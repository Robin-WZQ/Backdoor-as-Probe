"""Canonical BaP correction: one random start, TTC escape, then probe repair.

The deployed method is intentionally single-path:

* one random initialization (seed 123), with no identity/candidate set;
* two projected ascent steps on raw CLIP image-feature drift;
* one projected descent step on the probe residual/benign-direction loss;
* one fixed final output, always projected around the input image at 4/255.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from ..core.clip import (
    IMAGE_EXTS,
    build_image_transform,
    iter_batches,
    load_clip_model,
    normalize_pixels,
)
from ..core.config import (
    CORRECTION_BENIGN_DIRECTION_WEIGHT,
    CORRECTION_CANDIDATE_SELECTION,
    CORRECTION_DIRECTION_POLICY,
    CORRECTION_EPSILON,
    CORRECTION_ESCAPE_STEPS,
    CORRECTION_METHOD_ID,
    CORRECTION_OUTPUT_RULE,
    CORRECTION_RANDOM_STARTS,
    CORRECTION_REPAIR_STEPS,
    CORRECTION_RESIDUAL_WEIGHT,
    CORRECTION_SEED,
    CORRECTION_STEP_SIZE,
)
from .. import enforce_gpu_policy
from ..core.protocol import IMAGE_SERIALIZATION


EPSILON = CORRECTION_EPSILON
STEP_SIZE = CORRECTION_STEP_SIZE
ESCAPE_STEPS = CORRECTION_ESCAPE_STEPS
REPAIR_STEPS = CORRECTION_REPAIR_STEPS
RANDOM_STARTS = CORRECTION_RANDOM_STARTS
RESIDUAL_WEIGHT = CORRECTION_RESIDUAL_WEIGHT
BENIGN_DIRECTION_WEIGHT = CORRECTION_BENIGN_DIRECTION_WEIGHT
DIRECTION_POLICY = CORRECTION_DIRECTION_POLICY
SEED = CORRECTION_SEED
METHOD_ID = CORRECTION_METHOD_ID
OUTPUT_RULE = CORRECTION_OUTPUT_RULE
CANDIDATE_SELECTION = CORRECTION_CANDIDATE_SELECTION
IMAGE_SERIALIZATION_LINF_SAFE = (
    "round_to_uint8_then_discrete_linf_reprojection"
)


@dataclass
class Projection:
    mean: torch.Tensor
    basis: torch.Tensor


@dataclass
class DirectionBank:
    benign: dict[int, torch.Tensor]
    benign_concentration: dict[int, float]


def image_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(
        (
            item
            for item in root.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda item: item.name,
    )
    if not paths:
        raise ValueError(f"No images found under {root}")
    return paths


class CorrectionImageDataset:
    """Unlabelled image directory used by the canonical correction path."""

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int,
        max_images: int | None = None,
    ) -> None:
        paths = image_paths(image_dir)
        self.paths = paths[:max_images] if max_images is not None else paths
        if not self.paths:
            raise ValueError("No correction images remain after applying the limit")
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name


class BenignDirectionDataset:
    """Unlabelled benign images used to build the frozen direction artifact."""

    def __init__(
        self,
        benign_dir: str | Path,
        image_size: int,
        max_images: int | None = None,
    ) -> None:
        paths = image_paths(benign_dir)
        self.paths = paths[:max_images] if max_images is not None else paths
        if not self.paths:
            raise ValueError("No benign images remain after applying the limit")
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        benign = Image.open(path).convert("RGB")
        return self.transform(benign), path.name


def get_mlp_fc2_module(loaded, layer: int):
    layers = loaded.model.vision_model.encoder.layers
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"mlp_layer must lie in [0, {len(layers) - 1}]")
    return layers[layer].mlp.fc2


def collect_fc2_inputs(
    loaded,
    pixels: torch.Tensor,
    mlp_layers: list[int],
    token_index: int,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(layer: int):
        def hook(_module, inputs):
            hidden = inputs[0]
            if token_index < 0 or token_index >= hidden.shape[1]:
                raise ValueError(
                    f"token_index must lie in [0, {hidden.shape[1] - 1}]"
                )
            captured[layer] = hidden[:, token_index, :]

        return hook

    for layer in mlp_layers:
        handles.append(
            get_mlp_fc2_module(loaded, layer).register_forward_pre_hook(
                hook_for(layer)
            )
        )
    try:
        loaded.model.vision_model(
            pixel_values=normalize_pixels(pixels.to(loaded.device))
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in mlp_layers if layer not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture fc2 inputs for layers {missing}")
    return captured


def load_clean_feature_map(
    path: str | Path, mlp_layers: list[int]
) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        raw_feature_map = payload.get("features", payload)
        if not isinstance(raw_feature_map, dict):
            raise TypeError("clean_features_path['features'] must be a layer map")
        feature_map = {int(layer): value for layer, value in raw_feature_map.items()}
    elif isinstance(payload, torch.Tensor) and len(mlp_layers) == 1:
        feature_map = {mlp_layers[0]: payload}
    else:
        raise TypeError(
            "clean_features_path must contain {layer: tensor}, or one tensor "
            "when a single layer is active"
        )
    missing = [layer for layer in mlp_layers if layer not in feature_map]
    if missing:
        raise ValueError(f"Clean features are missing layers {missing}")
    return {layer: feature_map[layer] for layer in mlp_layers}


def build_projection(
    clean: torch.Tensor,
    rank: int | None,
    center: bool,
    svd_eps: float,
) -> tuple[Projection, int]:
    clean = clean.float()
    mean = clean.mean(dim=0) if center else clean.new_zeros(clean.shape[1])
    matrix = clean - mean.view(1, -1) if center else clean
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    effective_rank = int(
        (singular_values > singular_values.max() * svd_eps).sum().item()
    )
    if rank is not None:
        effective_rank = min(effective_rank, rank)
    if effective_rank <= 0:
        raise ValueError("Projection rank collapsed to zero")
    return (
        Projection(mean=mean, basis=vh[:effective_rank].T.contiguous()),
        effective_rank,
    )


def build_projections(args, device: torch.device):
    feature_map = load_clean_feature_map(
        args.clean_features_path, args.mlp_layers
    )
    projections: dict[int, Projection] = {}
    ranks: dict[int, int] = {}
    for layer in args.mlp_layers:
        projection, effective_rank = build_projection(
            feature_map[layer].to(device),
            args.rank,
            args.center,
            args.svd_eps,
        )
        projections[layer] = Projection(
            mean=projection.mean.to(device), basis=projection.basis.to(device)
        )
        ranks[layer] = effective_rank
    return projections, ranks


def residual_and_direction(
    activation: torch.Tensor,
    projection: Projection,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = activation - projection.mean.view(1, -1)
    projected = (
        centered @ projection.basis @ projection.basis.T
        + projection.mean.view(1, -1)
    )
    residual = activation - projected
    norm = residual.norm(p=2, dim=1)
    direction = residual / norm.clamp_min(eps).view(-1, 1)
    return residual, norm, direction


def mean_direction(values: list[torch.Tensor]):
    usable = [value for value in values if value.shape[0] > 0]
    if not usable:
        raise ValueError("No usable benign residual directions were found")
    vector = torch.cat(usable, dim=0).mean(dim=0)
    concentration = float(vector.norm(p=2).item())
    return F.normalize(vector, dim=0), concentration


def build_direction_bank(
    loaded,
    dataset: BenignDirectionDataset,
    projections: dict[int, Projection],
    args,
) -> DirectionBank:
    benign_directions = {layer: [] for layer in args.mlp_layers}
    total = math.ceil(len(dataset) / args.batch_size)
    with torch.no_grad():
        for items in tqdm(
            iter_batches(dataset, args.batch_size),
            total=total,
            desc="Building frozen direction bank",
        ):
            benign = torch.stack([item[0] for item in items]).to(loaded.device)
            benign_activations = collect_fc2_inputs(
                loaded, benign, args.mlp_layers, args.token_index
            )
            for layer in args.mlp_layers:
                _, benign_norm, benign_direction = residual_and_direction(
                    benign_activations[layer],
                    projections[layer],
                    args.direction_eps,
                )
                keep = benign_norm > args.direction_eps
                benign_directions[layer].append(benign_direction[keep].detach())
    benign_mean: dict[int, torch.Tensor] = {}
    benign_concentration: dict[int, float] = {}
    for layer in args.mlp_layers:
        benign_mean[layer], benign_concentration[layer] = mean_direction(
            benign_directions[layer]
        )
    return DirectionBank(
        benign=benign_mean,
        benign_concentration=benign_concentration,
    )


def load_direction_bank(path: str | Path, device: torch.device) -> DirectionBank:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"benign", "benign_concentration"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Invalid direction artifact: {path}")
    return DirectionBank(
        benign={int(k): value.to(device) for k, value in payload["benign"].items()},
        benign_concentration={
            int(k): float(value)
            for k, value in payload["benign_concentration"].items()
        },
    )


def probe_repair_loss_values(
    loaded,
    corrected: torch.Tensor,
    projections: dict[int, Projection],
    directions: DirectionBank,
    mlp_layers: list[int],
    token_index: int,
    direction_eps: float,
    *,
    residual_weight: float = RESIDUAL_WEIGHT,
    benign_direction_weight: float = BENIGN_DIRECTION_WEIGHT,
) -> torch.Tensor:
    activations = collect_fc2_inputs(
        loaded, corrected, mlp_layers, token_index
    )
    layer_losses = []
    for layer, activation in activations.items():
        residual, _, direction = residual_and_direction(
            activation, projections[layer], direction_eps
        )
        residual_sq = residual.square().sum(dim=1)
        benign_alignment = direction @ directions.benign[layer]
        layer_losses.append(
            residual_weight * residual_sq
            - benign_direction_weight * benign_alignment
        )
    return torch.stack(layer_losses).mean(dim=0)


def probe_suppression_loss_values(
    loaded,
    corrected: torch.Tensor,
    delta_weights: dict[int, torch.Tensor],
    mlp_layers: list[int],
    token_index: int,
) -> torch.Tensor:
    """Return the per-image probe-deactivation loss.

    ``delta_weights[layer]`` is the actual edited-minus-base ``fc2`` weight
    difference.  The loss is therefore exactly

    ``||Delta W_l h'_l(corrected)||_2^2``

    and does not use an attack direction or a classification label.
    """

    activations = collect_fc2_inputs(loaded, corrected, mlp_layers, token_index)
    layer_losses = []
    for layer in mlp_layers:
        if layer not in delta_weights:
            raise KeyError(f"delta_weights is missing layer {layer}")
        response = activations[layer] @ delta_weights[layer].to(
            device=activations[layer].device,
            dtype=activations[layer].dtype,
        ).T
        layer_losses.append(response.square().sum(dim=1))
    return torch.stack(layer_losses).mean(dim=0)


def raw_image_features(loaded, pixels: torch.Tensor) -> torch.Tensor:
    return loaded.model.get_image_features(
        pixel_values=normalize_pixels(pixels.to(loaded.device))
    )


def feature_drift(
    features: torch.Tensor, anchor_features: torch.Tensor
) -> torch.Tensor:
    return (features - anchor_features).square().sum(dim=1)


def correct_batch(
    loaded,
    input_pixels: torch.Tensor,
    projections: dict[int, Projection],
    directions: DirectionBank,
    *,
    mlp_layers: list[int],
    token_index: int,
    direction_eps: float,
    escape_steps: int = ESCAPE_STEPS,
    repair_steps: int = REPAIR_STEPS,
    residual_weight: float = RESIDUAL_WEIGHT,
    benign_direction_weight: float = BENIGN_DIRECTION_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a fixed escape-then-repair trajectory.

    The CLI calls this function only with the canonical defaults. Explicit
    overrides are available for controlled ablations and are not runtime
    method options.
    """

    if escape_steps < 0 or repair_steps < 0:
        raise ValueError("escape_steps and repair_steps must be non-negative")

    input_pixels = input_pixels.to(loaded.device)
    lower = (input_pixels - EPSILON).clamp(0.0, 1.0)
    upper = (input_pixels + EPSILON).clamp(0.0, 1.0)
    with torch.no_grad():
        anchor_features = raw_image_features(loaded, input_pixels).detach()

    current = input_pixels + torch.empty_like(input_pixels).uniform_(
        -EPSILON, EPSILON
    )
    current = current.clamp(0.0, 1.0).maximum(lower).minimum(upper)
    with torch.no_grad():
        initial_drift = feature_drift(
            raw_image_features(loaded, current), anchor_features
        )

    for _ in range(escape_steps):
        point = current.detach().requires_grad_(True)
        drift = feature_drift(raw_image_features(loaded, point), anchor_features)
        gradient = torch.autograd.grad(drift.sum(), point, only_inputs=True)[0]
        current = point.detach() + STEP_SIZE * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)

    with torch.no_grad():
        escaped_drift = feature_drift(
            raw_image_features(loaded, current), anchor_features
        )
        initial_repair_loss = probe_repair_loss_values(
            loaded,
            current,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        )

    for _ in range(repair_steps):
        point = current.detach().requires_grad_(True)
        repair_loss = probe_repair_loss_values(
            loaded,
            point,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        )
        gradient = torch.autograd.grad(
            repair_loss.sum(), point, only_inputs=True
        )[0]
        current = point.detach() - STEP_SIZE * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)

    with torch.no_grad():
        final_repair_loss = probe_repair_loss_values(
            loaded,
            current,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        )
        final_drift = feature_drift(
            raw_image_features(loaded, current), anchor_features
        )
        linf = (current - input_pixels).flatten(1).abs().max(dim=1).values
    return current.detach(), {
        "escape_initial_drift": initial_drift.detach().cpu(),
        "escape_final_drift": escaped_drift.detach().cpu(),
        "repair_initial_loss": initial_repair_loss.detach().cpu(),
        "repair_final_loss": final_repair_loss.detach().cpu(),
        "final_ttc_drift": final_drift.detach().cpu(),
        "linf_to_input": linf.detach().cpu(),
    }


def correct_batch_probe_suppression_then_repair(
    loaded,
    input_pixels: torch.Tensor,
    projections: dict[int, Projection],
    directions: DirectionBank,
    delta_weights: dict[int, torch.Tensor],
    *,
    mlp_layers: list[int],
    token_index: int,
    direction_eps: float,
    suppression_steps: int = 2,
    repair_steps: int = 1,
    suppression_step_size: float = STEP_SIZE,
    repair_step_size: float = STEP_SIZE,
    residual_weight: float = RESIDUAL_WEIGHT,
    benign_direction_weight: float = BENIGN_DIRECTION_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run two probe-suppression steps followed by the unchanged R1 loss.

    This is an experimental E2R1 variant.  The original E2R1 R1 objective is
    delegated unchanged to :func:`probe_repair_loss_values`; only the E2 stage
    is replaced by two projected descent steps on the actual edit response.
    """

    if suppression_steps < 0 or repair_steps < 0:
        raise ValueError("suppression_steps and repair_steps must be non-negative")
    if not mlp_layers:
        raise ValueError("mlp_layers must be non-empty")

    input_pixels = input_pixels.to(loaded.device)
    lower = (input_pixels - EPSILON).clamp(0.0, 1.0)
    upper = (input_pixels + EPSILON).clamp(0.0, 1.0)

    current = input_pixels + torch.empty_like(input_pixels).uniform_(
        -EPSILON, EPSILON
    )
    current = current.clamp(0.0, 1.0).maximum(lower).minimum(upper)

    with torch.no_grad():
        suppression_initial = probe_suppression_loss_values(
            loaded, current, delta_weights, mlp_layers, token_index
        ).detach()

    suppression_history = [suppression_initial]
    for _ in range(suppression_steps):
        point = current.detach().requires_grad_(True)
        suppression_loss = probe_suppression_loss_values(
            loaded, point, delta_weights, mlp_layers, token_index
        )
        gradient = torch.autograd.grad(
            suppression_loss.sum(), point, only_inputs=True
        )[0]
        current = point.detach() - float(suppression_step_size) * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)
        with torch.no_grad():
            suppression_history.append(
                probe_suppression_loss_values(
                    loaded, current, delta_weights, mlp_layers, token_index
                ).detach()
            )

    with torch.no_grad():
        repair_initial = probe_repair_loss_values(
            loaded,
            current,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        ).detach()

    for _ in range(repair_steps):
        point = current.detach().requires_grad_(True)
        repair_loss = probe_repair_loss_values(
            loaded,
            point,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        )
        gradient = torch.autograd.grad(
            repair_loss.sum(), point, only_inputs=True
        )[0]
        current = point.detach() - float(repair_step_size) * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)

    with torch.no_grad():
        suppression_final = probe_suppression_loss_values(
            loaded, current, delta_weights, mlp_layers, token_index
        ).detach()
        repair_final = probe_repair_loss_values(
            loaded,
            current,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        ).detach()
        linf = (current - input_pixels).flatten(1).abs().max(dim=1).values

    diagnostics: dict[str, torch.Tensor] = {
        "probe_suppression_initial": suppression_history[0].cpu(),
        "probe_suppression_after_step1": suppression_history[
            min(1, len(suppression_history) - 1)
        ].cpu(),
        "probe_suppression_after_step2": suppression_history[
            min(2, len(suppression_history) - 1)
        ].cpu(),
        "probe_suppression_final": suppression_final.cpu(),
        "repair_initial_loss": repair_initial.cpu(),
        "repair_final_loss": repair_final.cpu(),
        "linf_to_input": linf.detach().cpu(),
    }
    return current.detach(), diagnostics


def correct_batch_e2r1_plus_probe(
    loaded,
    input_pixels: torch.Tensor,
    projections: dict[int, Projection],
    directions: DirectionBank,
    delta_weights: dict[int, torch.Tensor],
    *,
    mlp_layers: list[int],
    token_index: int,
    direction_eps: float,
    escape_steps: int = ESCAPE_STEPS,
    probe_steps: int = 2,
    repair_steps: int = REPAIR_STEPS,
    escape_step_size: float = STEP_SIZE,
    probe_step_size: float = 0.05 / 255.0,
    repair_step_size: float = STEP_SIZE,
    residual_weight: float = RESIDUAL_WEIGHT,
    benign_direction_weight: float = BENIGN_DIRECTION_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep the original E2R1 escape, then add small-step probe suppression.

    The two feature-drift escape updates and the R1 repair objective follow
    :func:`correct_batch` exactly.  The extra probe updates are inserted
    between them and optimize the actual edited-minus-base ``fc2`` response.
    This helper is experimental and is not used by the canonical entry point.
    """

    if escape_steps < 0 or probe_steps < 0 or repair_steps < 0:
        raise ValueError("step counts must be non-negative")
    if not mlp_layers:
        raise ValueError("mlp_layers must be non-empty")

    input_pixels = input_pixels.to(loaded.device)
    lower = (input_pixels - EPSILON).clamp(0.0, 1.0)
    upper = (input_pixels + EPSILON).clamp(0.0, 1.0)
    current = input_pixels + torch.empty_like(input_pixels).uniform_(
        -EPSILON, EPSILON
    )
    current = current.clamp(0.0, 1.0).maximum(lower).minimum(upper)

    with torch.no_grad():
        anchor_features = raw_image_features(loaded, input_pixels).detach()
        escape_initial = feature_drift(
            raw_image_features(loaded, current), anchor_features
        ).detach()
        probe_before_escape = probe_suppression_loss_values(
            loaded, current, delta_weights, mlp_layers, token_index
        ).detach()

    escape_history = [escape_initial]
    for _ in range(escape_steps):
        point = current.detach().requires_grad_(True)
        drift = feature_drift(raw_image_features(loaded, point), anchor_features)
        gradient = torch.autograd.grad(drift.sum(), point, only_inputs=True)[0]
        current = point.detach() + float(escape_step_size) * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)
        with torch.no_grad():
            escape_history.append(
                feature_drift(raw_image_features(loaded, current), anchor_features)
                .detach()
            )

    with torch.no_grad():
        probe_initial = probe_suppression_loss_values(
            loaded, current, delta_weights, mlp_layers, token_index
        ).detach()

    probe_history = [probe_initial]
    for _ in range(probe_steps):
        point = current.detach().requires_grad_(True)
        probe_loss = probe_suppression_loss_values(
            loaded, point, delta_weights, mlp_layers, token_index
        )
        gradient = torch.autograd.grad(probe_loss.sum(), point, only_inputs=True)[0]
        current = point.detach() - float(probe_step_size) * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)
        with torch.no_grad():
            probe_history.append(
                probe_suppression_loss_values(
                    loaded, current, delta_weights, mlp_layers, token_index
                ).detach()
            )

    with torch.no_grad():
        repair_initial = probe_repair_loss_values(
            loaded,
            current,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        ).detach()

    for _ in range(repair_steps):
        point = current.detach().requires_grad_(True)
        repair_loss = probe_repair_loss_values(
            loaded,
            point,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        )
        gradient = torch.autograd.grad(
            repair_loss.sum(), point, only_inputs=True
        )[0]
        current = point.detach() - float(repair_step_size) * gradient.sign()
        current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)

    with torch.no_grad():
        probe_final = probe_suppression_loss_values(
            loaded, current, delta_weights, mlp_layers, token_index
        ).detach()
        repair_final = probe_repair_loss_values(
            loaded,
            current,
            projections,
            directions,
            mlp_layers,
            token_index,
            direction_eps,
            residual_weight=residual_weight,
            benign_direction_weight=benign_direction_weight,
        ).detach()
        final_drift = feature_drift(
            raw_image_features(loaded, current), anchor_features
        ).detach()
        linf = (current - input_pixels).flatten(1).abs().max(dim=1).values

    def at_or_last(history: list[torch.Tensor], index: int) -> torch.Tensor:
        return history[min(index, len(history) - 1)].cpu()

    diagnostics: dict[str, torch.Tensor] = {
        "escape_initial_drift": at_or_last(escape_history, 0),
        "escape_after_step1_drift": at_or_last(escape_history, 1),
        "escape_after_step2_drift": at_or_last(escape_history, 2),
        "probe_suppression_before_escape": probe_before_escape.cpu(),
        "probe_suppression_initial": at_or_last(probe_history, 0),
        "probe_suppression_after_step1": at_or_last(probe_history, 1),
        "probe_suppression_after_step2": at_or_last(probe_history, 2),
        "probe_suppression_final": probe_final.cpu(),
        "repair_initial_loss": repair_initial.cpu(),
        "repair_final_loss": repair_final.cpu(),
        "final_ttc_drift": final_drift.cpu(),
        "linf_to_input": linf.detach().cpu(),
    }
    return current.detach(), diagnostics


def save_linf_safe_image(
    tensor: torch.Tensor,
    anchor: torch.Tensor,
    path: Path,
) -> None:
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    anchor = anchor.detach().clamp(0.0, 1.0).cpu()
    if tensor.shape != anchor.shape:
        raise ValueError(
            f"Tensor/anchor shape mismatch: {tuple(tensor.shape)} vs "
            f"{tuple(anchor.shape)}"
        )
    scale = 255.0
    tolerance = 1e-4
    candidate = (tensor * scale).round().to(torch.int16)
    lower = (
        ((anchor - EPSILON).clamp(0.0, 1.0) * scale - tolerance)
        .ceil()
        .to(torch.int16)
    )
    upper = (
        ((anchor + EPSILON).clamp(0.0, 1.0) * scale + tolerance)
        .floor()
        .to(torch.int16)
    )
    candidate = candidate.maximum(lower).minimum(upper).clamp(0, 255)
    to_pil_image(candidate.to(torch.uint8)).save(path)


def tensor_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited_model_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--clean_features_path", required=True)
    parser.add_argument("--global_direction_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--mlp_layer", type=int, default=6)
    parser.add_argument("--token_index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--svd_eps", type=float, default=1e-6)
    parser.add_argument("--direction_eps", type=float, default=1e-8)
    parser.add_argument(
        "--escape_steps",
        type=int,
        default=ESCAPE_STEPS,
        help="Controlled-ablation override; the deployed default remains canonical E2.",
    )
    parser.add_argument(
        "--repair_steps",
        type=int,
        default=REPAIR_STEPS,
        help="Controlled-ablation override; the deployed default remains canonical R1.",
    )
    parser.add_argument(
        "--residual_weight",
        type=float,
        default=RESIDUAL_WEIGHT,
        help="Controlled-ablation override for the probe residual term.",
    )
    parser.add_argument(
        "--benign_direction_weight",
        type=float,
        default=BENIGN_DIRECTION_WEIGHT,
        help="Controlled-ablation override for benign residual-direction guidance.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_gpu_policy(args.device)
    if args.escape_steps < 0 or args.repair_steps < 0:
        raise ValueError("escape_steps and repair_steps must be non-negative")
    if args.residual_weight < 0 or args.benign_direction_weight < 0:
        raise ValueError("objective weights must be non-negative")
    args.mlp_layers = [args.mlp_layer]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_clip_model(
        args.edited_model_dir,
        args.device,
        local_files_only=args.local_files_only,
    )
    projections, ranks = build_projections(args, loaded.device)
    directions = load_direction_bank(
        args.global_direction_artifact, loaded.device
    )
    missing = [
        layer
        for layer in args.mlp_layers
        if layer not in directions.benign
    ]
    if missing:
        raise ValueError(f"Direction artifact is missing layers {missing}")
    dataset = CorrectionImageDataset(
        args.image_dir, args.image_size, args.max_images
    )

    rows: list[dict] = []
    metrics = {
        key: []
        for key in (
            "escape_initial_drift",
            "escape_final_drift",
            "repair_initial_loss",
            "repair_final_loss",
            "final_ttc_drift",
            "linf_to_input",
        )
    }
    total = math.ceil(len(dataset) / args.batch_size)
    for items in tqdm(
        iter_batches(dataset, args.batch_size),
        total=total,
        desc=f"BaP escape={args.escape_steps} repair={args.repair_steps}",
    ):
        inputs = torch.stack([item[0] for item in items]).to(loaded.device)
        names = [item[1] for item in items]
        corrected, diagnostics = correct_batch(
            loaded,
            inputs,
            projections,
            directions,
            mlp_layers=args.mlp_layers,
            token_index=args.token_index,
            direction_eps=args.direction_eps,
            escape_steps=args.escape_steps,
            repair_steps=args.repair_steps,
            residual_weight=args.residual_weight,
            benign_direction_weight=args.benign_direction_weight,
        )
        for key, values in diagnostics.items():
            metrics[key].append(values)
        for index, name in enumerate(names):
            save_linf_safe_image(
                corrected[index],
                inputs[index],
                image_dir / f"{Path(name).stem}.png",
            )
            rows.append(
                {
                    "image_name": name,
                    **{
                        key: float(values[index])
                        for key, values in diagnostics.items()
                    },
                }
            )

    csv_path = output_dir / "correction.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    is_canonical = (
        args.escape_steps == ESCAPE_STEPS
        and args.repair_steps == REPAIR_STEPS
        and args.residual_weight == RESIDUAL_WEIGHT
        and args.benign_direction_weight == BENIGN_DIRECTION_WEIGHT
    )
    summary = {
        "method": METHOD_ID if is_canonical else f"controlled_ablation_of_{METHOD_ID}",
        "configuration_role": "canonical" if is_canonical else "controlled_ablation",
        "parameters": {
            "random_starts": RANDOM_STARTS,
            "escape_steps": args.escape_steps,
            "repair_steps": args.repair_steps,
            "epsilon": EPSILON,
            "step_size": STEP_SIZE,
            "residual_weight": args.residual_weight,
            "benign_direction_weight": args.benign_direction_weight,
            "seed": SEED,
            "mlp_layer": args.mlp_layer,
            "token_index": args.token_index,
        },
        "protocol": {
            "stage_1": "maximize raw CLIP image-feature drift",
            "stage_2": "minimize probe residual and benign-direction loss",
            "output": f"fixed final iterate after repair step {args.repair_steps}",
            "output_rule": OUTPUT_RULE,
            "candidate_selection": CANDIDATE_SELECTION,
            "direction_policy": DIRECTION_POLICY,
            "optimization_uses_labels": False,
            "projection_center": "input image",
        },
        "image_serialization": IMAGE_SERIALIZATION_LINF_SAFE,
        "source_image_serialization": IMAGE_SERIALIZATION,
        "num_images": len(dataset),
        "image_dir": str(image_dir.resolve()),
        "correction_csv": str(csv_path.resolve()),
        "projection_ranks": ranks,
        "direction_source": str(
            Path(args.global_direction_artifact).resolve()
        ),
        "metrics": {
            key: tensor_summary(torch.cat(values))
            for key, values in metrics.items()
        },
    }
    summary_path = output_dir / "correction_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "images": len(dataset)}, indent=2))


if __name__ == "__main__":
    main()

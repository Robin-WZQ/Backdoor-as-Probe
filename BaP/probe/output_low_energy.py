"""Output-low-energy semantic BaP probe.

This module is the implantation and detector path.  It deliberately keeps
the input trigger and the output semantic target separate:

* ``t_l`` is supplied as a runtime tensor or rebuilt from an external paired
  attack-calibration directory.  A clean-only fallback exists behind an
  explicit smoke-check flag; no attack images are bundled in this repository.
* ``y`` is constructed only from benign calibration data.  If ``H`` contains
  the clean ``fc2`` inputs and ``W`` is the *base* ``fc2`` weight, we form
  ``Z = H W^T`` and project ``A^dagger q`` onto the bottom right-singular
  subspace of ``Z``.
* The edit is ``Delta W = y t^T / ||t||^2``.  Detection reads the edited
  matrix output directly, using the exact signed bias-free score
  ``p_y(x) = y_hat^T W' h_l(x)``.

The frozen defaults are ``K_in=256`` for the input trigger subspace,
``K_out=32`` for the output target subspace, clean q95 calibration,
``trigger_norm=5`` and ``target_scale=40``.

The public correction implementation is intentionally elsewhere
(``BaP.correction.current``) and is not changed by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import enforce_gpu_policy
from ..core.clip import (
    DirectoryImageDataset,
    build_image_transform,
    encode_text_features,
    load_clip_model,
    normalize_pixels,
    set_seed,
)


# Default probe parameters. The input and
# output ranks refer to separate low-energy subspaces; they do not change the
# physical fc2 dimensions (3072 input, 768 output).
DEFAULT_QUANTILE = 0.95
DEFAULT_LAYER = 6
DEFAULT_TOKEN = 0
DEFAULT_INPUT_LOW_ENERGY_RANK = 256
DEFAULT_OUTPUT_LOW_ENERGY_RANK = 32
DEFAULT_TRIGGER_NORM = 5.0
DEFAULT_TARGET_SCALE = 40.0


def detector_artifact_name(quantile: float) -> str:
    """Return the stable artifact directory name for a calibration quantile.

    The historical naming convention drops the decimal point (q95,
    q975).  Keeping this in one helper prevents a custom calibration from
    accidentally writing a detector under the q95 directory.
    """

    percent = f"{100.0 * float(quantile):.3f}".rstrip("0").rstrip(".")
    return f"target_response_q{percent.replace('.', '')}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _limited_dataset(
    image_dir: str | Path,
    image_size: int,
    limit: int | None,
) -> DirectoryImageDataset:
    dataset = DirectoryImageDataset(
        str(image_dir), transform=build_image_transform(image_size)
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when supplied")
        dataset.image_paths = dataset.image_paths[:limit]
    if not dataset.image_paths:
        raise ValueError(f"No images remain under {image_dir}")
    return dataset


def collect_fc2_inputs(
    loaded: Any,
    image_dir: str | Path,
    *,
    mlp_layer: int = DEFAULT_LAYER,
    token_index: int = DEFAULT_TOKEN,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    limit: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Collect clean, unpatched ``fc2`` inputs ``H`` from a benign directory."""

    dataset = _limited_dataset(image_dir, image_size, limit)
    module = loaded.model.vision_model.encoder.layers[mlp_layer].mlp.fc2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs):
        hidden = inputs[0]
        if hidden.ndim != 3:
            raise RuntimeError(f"Expected [batch,tokens,dim] fc2 input, got {hidden.shape}")
        if token_index < 0 or token_index >= hidden.shape[1]:
            raise ValueError(
                f"token_index must be in [0, {hidden.shape[1] - 1}], got {token_index}"
            )
        captured["value"] = hidden[:, token_index, :].detach().float().cpu()

    handle = module.register_forward_pre_hook(hook)
    values: list[torch.Tensor] = []
    names: list[str] = []
    try:
        with torch.inference_mode():
            for pixels, batch_names in tqdm(
                loader, desc="Collecting benign fc2 inputs", leave=False
            ):
                loaded.model.vision_model(
                    pixel_values=normalize_pixels(
                        pixels.to(
                            loaded.device,
                            non_blocking=loaded.device.type == "cuda",
                        )
                    )
                )
                if "value" not in captured:
                    raise RuntimeError("fc2 input hook did not capture a batch")
                values.append(captured.pop("value"))
                names.extend(list(batch_names))
    finally:
        handle.remove()
    if not values:
        raise ValueError("No fc2 inputs were collected")
    return torch.cat(values, dim=0), names


def load_feature_cache(
    path: str | Path,
    *,
    image_dir: str | Path | None = None,
    mlp_layer: int = DEFAULT_LAYER,
) -> tuple[torch.Tensor, list[str] | None]:
    """Load a ``{layer: tensor}`` cache produced by this module."""

    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    names: list[str] | None = None
    if isinstance(payload, dict) and "features" in payload:
        raw = payload["features"]
        raw_names = payload.get("names")
        if raw_names is not None:
            names = [str(item) for item in raw_names]
    else:
        raw = payload
    if not isinstance(raw, dict):
        raise TypeError(f"Feature cache must be a layer map: {cache_path}")
    value = raw.get(mlp_layer, raw.get(str(mlp_layer)))
    if value is None:
        raise KeyError(f"Feature cache has no layer {mlp_layer}: {cache_path}")
    features = torch.as_tensor(value).float()
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError(f"Invalid feature tensor in {cache_path}: {features.shape}")
    if names is not None and len(names) != features.shape[0]:
        raise ValueError("Feature-cache names and rows have different lengths")
    if image_dir is not None:
        directory_names = sorted(
            item.name
            for item in Path(image_dir).iterdir()
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if names is None:
            if len(directory_names) != features.shape[0]:
                raise ValueError(
                    "Legacy feature cache has no names and its row count does "
                    "not match the benign image directory"
                )
            names = directory_names
        elif not {Path(name).stem for name in names}.issubset(
            {Path(name).stem for name in directory_names}
        ):
            raise ValueError("Feature cache does not match the benign image directory")
    return features, names


def _load_trigger(
    path: str | Path,
    *,
    mlp_layer: int,
    input_dim: int,
) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        value = payload.get(mlp_layer, payload.get(str(mlp_layer)))
        if value is None:
            for key in ("trigger", "t", "direction"):
                if key in payload:
                    value = payload[key]
                    break
        if value is None:
            raise KeyError(f"No trigger for layer {mlp_layer} in {path}")
    else:
        value = payload
    trigger = torch.as_tensor(value).float().flatten()
    if trigger.numel() != input_dim:
        raise ValueError(
            f"Trigger width {trigger.numel()} does not match fc2 input width {input_dim}"
        )
    if not torch.isfinite(trigger).all() or trigger.norm() <= 1e-12:
        raise ValueError("Trigger must be finite and non-zero")
    return trigger


def derive_clean_trigger(
    clean_h: torch.Tensor,
    *,
    trigger_norm: float = DEFAULT_TRIGGER_NORM,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Derive a deterministic clean-only fallback trigger.

    This fallback is provided so this implementation can be exercised without attack data.  For
    reproducing an attack-aligned probe, pass the corresponding trigger file
    through ``--trigger_path`` instead.
    """

    if trigger_norm <= 0:
        raise ValueError("trigger_norm must be positive")
    _, singular_values, vh = torch.linalg.svd(clean_h.float(), full_matrices=False)
    if vh.shape[0] == 0:
        raise ValueError("Cannot derive a trigger from an empty feature matrix")
    trigger = vh[-1].clone()
    trigger = trigger / trigger.norm().clamp_min(1e-12) * float(trigger_norm)
    return trigger, {
        "source": "clean_bottom_right_singular_vector",
        "singular_value": float(singular_values[-1]),
        "paper_protocol": False,
    }


def build_attack_aligned_trigger(
    clean_h: torch.Tensor,
    clean_names: list[str],
    attack_h: torch.Tensor,
    attack_names: list[str],
    *,
    input_low_energy_rank: int = DEFAULT_INPUT_LOW_ENERGY_RANK,
    trigger_norm: float = DEFAULT_TRIGGER_NORM,
    whitening_floor: float = 0.01,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct the paper-protocol input trigger from paired development data.

    The attack directory is an external runtime input and is never bundled in
    BaP.  Rows are aligned by image stem before the clean-to-attack shift is
    projected into the clean input-side bottom singular subspace and whitened.
    """

    if clean_h.ndim != 2 or attack_h.ndim != 2 or clean_h.shape[1] != attack_h.shape[1]:
        raise ValueError("Clean and attack feature matrices must have the same width")
    if len(clean_names) != clean_h.shape[0] or len(attack_names) != attack_h.shape[0]:
        raise ValueError("Feature rows and image-name counts must agree")
    if input_low_energy_rank <= 0 or trigger_norm <= 0 or whitening_floor < 0:
        raise ValueError("Invalid trigger-construction hyperparameters")

    clean_stems = [Path(name).stem for name in clean_names]
    attack_stems = [Path(name).stem for name in attack_names]
    if len(set(clean_stems)) != len(clean_stems) or len(set(attack_stems)) != len(attack_stems):
        raise ValueError("Clean/attack calibration image stems must be unique")
    attack_index = {name: index for index, name in enumerate(attack_stems)}
    missing = [name for name in clean_stems if name not in attack_index]
    extra = [name for name in attack_stems if name not in set(clean_stems)]
    if missing or extra:
        raise ValueError(
            f"Clean/attack calibration sets differ: missing={len(missing)}, extra={len(extra)}"
        )
    order = torch.tensor([attack_index[name] for name in clean_stems], dtype=torch.long)
    aligned_attack = attack_h[order].float()
    clean = clean_h.float()

    _, singular_values, vh = torch.linalg.svd(clean, full_matrices=False)
    rank = min(int(input_low_energy_rank), int(vh.shape[0]))
    bottom_values = singular_values[-rank:]
    bottom_basis = vh[-rank:]
    shift = (aligned_attack - clean).mean(dim=0)
    coefficients = bottom_basis @ shift
    scale = bottom_values.square().mean().clamp_min(1e-12)
    coefficients = coefficients / (
        bottom_values.square() + float(whitening_floor) * scale
    )
    direction = (coefficients[:, None] * bottom_basis).sum(dim=0)
    if direction.norm() <= 1e-12:
        raise ValueError("The paired attack shift is numerically zero after projection")
    trigger = direction / direction.norm() * float(trigger_norm)
    clean_projection = clean @ trigger
    attack_projection = aligned_attack @ trigger
    return trigger, {
        "source": "paired_attack_aligned_whitened_mean_shift",
        "paper_protocol": True,
        "paired_count": int(clean.shape[0]),
        "input_low_energy_rank": int(rank),
        "whitening_floor": float(whitening_floor),
        "clean_projection_abs_mean": float(clean_projection.abs().mean()),
        "attack_projection_abs_mean": float(attack_projection.abs().mean()),
        "paired_shift_projection": float(shift @ trigger),
    }


def build_target_from_clean_output(
    clean_h: torch.Tensor,
    base_weight: torch.Tensor,
    visual_projection: torch.Tensor,
    text_feature: torch.Tensor,
    *,
    output_low_energy_rank: int = DEFAULT_OUTPUT_LOW_ENERGY_RANK,
    target_scale: float = DEFAULT_TARGET_SCALE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Construct ``y`` by projecting ``A^dagger q`` into output low energy."""

    if clean_h.ndim != 2 or base_weight.ndim != 2:
        raise ValueError("clean_h and base_weight must be matrices")
    if clean_h.shape[1] != base_weight.shape[1]:
        raise ValueError("clean_h width does not match fc2 input width")
    if output_low_energy_rank <= 0:
        raise ValueError("output_low_energy_rank must be positive")
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")

    clean_output = clean_h.float() @ base_weight.float().T
    _, singular_values, vh = torch.linalg.svd(clean_output, full_matrices=False)
    rank = min(int(output_low_energy_rank), int(vh.shape[0]))
    if rank <= 0:
        raise ValueError("The clean output matrix has no usable singular directions")
    low_basis = vh[-rank:]

    projection = visual_projection.float().cpu()
    q = F.normalize(text_feature.float().cpu(), dim=0)
    semantic_target = torch.linalg.pinv(projection) @ q
    semantic_target = semantic_target.float()
    semantic_unit = semantic_target / semantic_target.norm().clamp_min(1e-12)
    projected = low_basis.T @ (low_basis @ semantic_unit)
    projected_norm = projected.norm()
    if projected_norm <= 1e-12:
        raise ValueError("Projected semantic target is numerically zero")
    y = projected / projected_norm * float(target_scale)
    y_hat = y / y.norm().clamp_min(1e-12)
    meta = {
        "output_matrix_shape": list(clean_output.shape),
        "output_low_energy_rank": int(rank),
        "clean_output_singular_max": float(singular_values.max()),
        "clean_output_singular_min": float(singular_values.min()),
        "semantic_target_norm_before_projection": float(semantic_target.norm()),
        "projected_target_norm_ratio": float(projected_norm),
        "projected_target_cosine_to_semantic": float(torch.dot(projected, semantic_unit) / projected_norm),
        "low_basis": low_basis,
        "singular_values": singular_values,
        "semantic_target_before_projection": semantic_target,
        "clean_output": clean_output,
        "clean_base_projection": clean_output @ y_hat,
    }
    return y, y_hat, meta


def calibrate_target_detector(
    clean_projection: torch.Tensor,
    *,
    quantile: float = DEFAULT_QUANTILE,
    edited_model_dir: str | Path | None = None,
    delta_y_path: str | Path | None = None,
    mlp_layer: int = DEFAULT_LAYER,
    token_index: int = DEFAULT_TOKEN,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Fit a clean-only high-response threshold on the raw signed response."""

    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must lie in (0.5, 1)")
    values = clean_projection.float().flatten()
    if values.numel() == 0:
        raise ValueError("Cannot calibrate an empty clean response set")
    scores = values
    threshold = torch.quantile(scores, float(quantile))
    detector: dict[str, Any] = {
        "version": 2,
        "method": "BaP output-low-energy target-direction response",
        "score_mode": "target_semantic_fc2_bias_free_signed",
        "score_definition": "y_hat dot W_prime dot h_l(x)",
        "score_scale": "raw_signed_response",
        "bias_included": False,
        "domain_policy": "one clean-only scalar calibration; no attack samples used",
        "calibration_count": int(values.numel()),
        "threshold_mode": "clean_empirical_quantile",
        "threshold_quantile": float(quantile),
        "threshold": float(threshold),
        "mlp_layer": int(mlp_layer),
        "token_index": int(token_index),
    }
    if edited_model_dir is not None:
        detector["edited_model_dir"] = str(Path(edited_model_dir).resolve())
    if delta_y_path is not None:
        detector["delta_y_path"] = str(Path(delta_y_path).resolve())
    return detector, scores


def score_target_values(
    loaded: Any,
    image_dir: str | Path,
    y_hat: torch.Tensor,
    *,
    mlp_layer: int = DEFAULT_LAYER,
    token_index: int = DEFAULT_TOKEN,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    limit: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Compute ``p_y`` from an edited model on an arbitrary image directory."""

    dataset = _limited_dataset(image_dir, image_size, limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    module = loaded.model.vision_model.encoder.layers[mlp_layer].mlp.fc2
    target = y_hat.to(loaded.device).float().flatten()
    target = target / target.norm().clamp_min(1e-12)
    captured: dict[str, torch.Tensor] = {}
    values: list[torch.Tensor] = []

    def hook(_module, inputs):
        hidden = inputs[0]
        if token_index < 0 or token_index >= hidden.shape[1]:
            raise ValueError(
                f"token_index must be in [0, {hidden.shape[1] - 1}], got {token_index}"
            )
        hidden = hidden[:, token_index, :]
        weight = _module.weight
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("Captured fc2 width does not match edited weight")
        # Compute the score inside the hook so this routine retains only a
        # scalar per image.  Keeping the full [B, tokens, 3072] activation
        # until the vision forward returns can exceed GPU memory at the
        # detector's intended batch sizes.
        response = (hidden.float() @ weight.float().T) @ target
        values.append(response.detach().cpu())
        captured["seen"] = torch.ones((), dtype=torch.uint8)

    handle = module.register_forward_pre_hook(hook)
    names: list[str] = []
    try:
        with torch.inference_mode():
            for pixels, batch_names in tqdm(
                loader, desc="Scoring target-direction response", leave=False
            ):
                loaded.model.vision_model(
                    pixel_values=normalize_pixels(
                        pixels.to(
                            loaded.device,
                            non_blocking=loaded.device.type == "cuda",
                        )
                    )
                )
                if "seen" not in captured:
                    raise RuntimeError("fc2 input hook did not capture a batch")
                captured.pop("seen")
                names.extend(list(batch_names))
    finally:
        handle.remove()
    return torch.cat(values), names


def _write_score_csv(
    path: Path,
    names: list[str],
    values: torch.Tensor,
    scores: torch.Tensor,
    detector: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = float(detector["threshold"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image_name", "score", "p_y", "threshold", "is_suspicious"]
        )
        for name, value, score in zip(names, values.tolist(), scores.tolist()):
            writer.writerow(
                [
                    name,
                    f"{score:.12g}",
                    f"{value:.12g}",
                    f"{threshold:.12g}",
                    int(score >= threshold),
                ]
            )


def _save_detector(detector: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detector.json").write_text(
        json.dumps(detector, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(torch.tensor(float(detector["threshold"]), dtype=torch.float32), output_dir / "threshold.pt")


def build_output_low_energy_probe(
    *,
    model_id: str,
    calibration_dir: str | Path,
    output_dir: str | Path,
    trigger_path: str | Path | None = None,
    attack_calibration_dir: str | Path | None = None,
    allow_clean_only_trigger: bool = False,
    probe_prompt: str = "a white teapot",
    mlp_layer: int = DEFAULT_LAYER,
    token_index: int = DEFAULT_TOKEN,
    output_low_energy_rank: int = DEFAULT_OUTPUT_LOW_ENERGY_RANK,
    input_low_energy_rank: int = DEFAULT_INPUT_LOW_ENERGY_RANK,
    whitening_floor: float = 0.01,
    trigger_norm: float = DEFAULT_TRIGGER_NORM,
    target_scale: float = DEFAULT_TARGET_SCALE,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    calibration_limit: int | None = None,
    device: str = "cuda:0",
    local_files_only: bool = False,
    feature_cache_path: str | Path | None = None,
    quantile: float = DEFAULT_QUANTILE,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Build and save an isolated output-low-energy probe artifact."""

    physical_gpu = enforce_gpu_policy(device)
    set_seed(seed)
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is non-empty: {output}; pass --force to overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_clip_model(model_id, device, local_files_only=local_files_only)
    fc2 = loaded.model.vision_model.encoder.layers[mlp_layer].mlp.fc2
    base_weight = fc2.weight.detach().float().cpu().clone()
    base_bias = fc2.bias.detach().float().cpu().clone() if fc2.bias is not None else None

    cache = Path(feature_cache_path).resolve() if feature_cache_path else output / f"clean_features_layers_{mlp_layer}_token_{token_index}.pt"
    if cache.is_file():
        clean_h, clean_names = load_feature_cache(cache, image_dir=calibration_dir, mlp_layer=mlp_layer)
        feature_source = str(cache)
    else:
        clean_h, clean_names = collect_fc2_inputs(
            loaded,
            calibration_dir,
            mlp_layer=mlp_layer,
            token_index=token_index,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            limit=calibration_limit,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": {mlp_layer: clean_h}, "names": clean_names}, cache
        )
        feature_source = "collected_online"

    if clean_names is None:
        raise RuntimeError("Clean feature rows have no image-name alignment")

    if clean_h.shape[1] != base_weight.shape[1]:
        raise ValueError("Clean fc2 input width does not match base fc2 weight")
    q = encode_text_features(loaded, [probe_prompt], prompt_template="{}")[0].detach().cpu()
    visual_projection = loaded.model.visual_projection.weight.detach().float().cpu()
    y, y_hat, target_meta = build_target_from_clean_output(
        clean_h,
        base_weight,
        visual_projection,
        q,
        output_low_energy_rank=output_low_energy_rank,
        target_scale=target_scale,
    )

    if trigger_path is not None and attack_calibration_dir is not None:
        raise ValueError(
            "Choose one input-trigger source: --trigger_path or "
            "--attack_calibration_dir, not both"
        )
    trigger_meta: dict[str, Any]
    if trigger_path is not None:
        trigger = _load_trigger(trigger_path, mlp_layer=mlp_layer, input_dim=base_weight.shape[1])
        trigger = trigger / trigger.norm().clamp_min(1e-12) * float(trigger_norm)
        trigger_meta = {
            "source": str(Path(trigger_path).resolve()),
            "mode": "external_trigger_artifact",
            "paper_protocol": True,
        }
    elif attack_calibration_dir is not None:
        attack_h, attack_names = collect_fc2_inputs(
            loaded,
            attack_calibration_dir,
            mlp_layer=mlp_layer,
            token_index=token_index,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            limit=calibration_limit,
        )
        trigger, trigger_meta = build_attack_aligned_trigger(
            clean_h,
            clean_names,
            attack_h,
            attack_names,
            input_low_energy_rank=input_low_energy_rank,
            trigger_norm=trigger_norm,
            whitening_floor=whitening_floor,
        )
        trigger_meta["attack_calibration_dir"] = str(
            Path(attack_calibration_dir).resolve()
        )
    elif allow_clean_only_trigger:
        trigger, trigger_meta = derive_clean_trigger(clean_h, trigger_norm=trigger_norm)
    else:
        raise ValueError(
            "Paper runs require --trigger_path or --attack_calibration_dir. "
            "Use --allow_clean_only_trigger only for a non-paper smoke check."
        )
    delta_weight = torch.outer(y, trigger) / trigger.square().sum().clamp_min(1e-12)
    edited_weight = base_weight + delta_weight

    clean_base_projection = target_meta["clean_base_projection"]
    clean_edited_projection = (clean_h @ edited_weight.T) @ y_hat
    detector, clean_scores = calibrate_target_detector(
        clean_edited_projection,
        quantile=quantile,
        edited_model_dir=output / "edited_model",
        delta_y_path=output / "delta_y.pt",
        mlp_layer=mlp_layer,
        token_index=token_index,
    )

    with torch.no_grad():
        fc2.weight.copy_(edited_weight.to(fc2.weight.device, dtype=fc2.weight.dtype))
    loaded.model.save_pretrained(output / "edited_model")
    loaded.processor.save_pretrained(output / "edited_model")
    torch.save({mlp_layer: trigger.cpu()}, output / "trigger.pt")
    torch.save(y.cpu(), output / "delta_y.pt")
    torch.save(delta_weight.cpu(), output / "delta_weight.pt")
    torch.save(
        {
            "layer": mlp_layer,
            "token_index": token_index,
            "rank": target_meta["output_low_energy_rank"],
            "basis": target_meta["low_basis"].cpu(),
            "singular_values": target_meta["singular_values"].cpu(),
            "semantic_target_before_projection": target_meta["semantic_target_before_projection"].cpu(),
        },
        output / "output_low_energy_basis.pt",
    )
    # The unchanged correction loader accepts this nested feature payload and
    # ignores the row-name metadata.
    torch.save(
        {"features": {mlp_layer: clean_h.cpu()}, "names": clean_names}, cache
    )
    detector_dir = output / detector_artifact_name(quantile)
    _save_detector(detector, detector_dir)
    calibration_csv = output / "calibration_scores.csv"
    _write_score_csv(calibration_csv, clean_names, clean_edited_projection, clean_scores, detector)
    # Keep a self-contained detector directory as well as the historical
    # root-level copy.  Downstream gate/evaluation scripts receive the
    # detector directory and should not need to know the implant layout.
    _write_score_csv(
        detector_dir / "calibration_scores.csv",
        clean_names,
        clean_edited_projection,
        clean_scores,
        detector,
    )
    (output / "calibration_summary.json").write_text(
        json.dumps(
            {
                "image_dir": str(Path(calibration_dir).resolve()),
                "count": len(clean_names),
                "p_y_mean": float(clean_edited_projection.mean()),
                "p_y_std": float(clean_edited_projection.std(unbiased=False)),
                "quantile": float(quantile),
                "threshold": detector["threshold"],
                "suspicious_rate": float((clean_scores >= detector["threshold"]).float().mean()),
                "csv": str(calibration_csv.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    model_file = output / "edited_model" / "model.safetensors"
    meta = {
        "variant": "output_low_energy_semantic_target",
        "base_model": str(Path(model_id).resolve()) if Path(model_id).exists() else model_id,
        "probe_prompt": probe_prompt,
        "mlp_layer": mlp_layer,
        "token_index": token_index,
        "calibration_dir": str(Path(calibration_dir).resolve()),
        "feature_source": feature_source,
        "clean_feature_count": int(clean_h.shape[0]),
        "semantic_target_formula": "A_pinv @ normalized_text_feature",
        "output_subspace_formula": "bottom right singular vectors of Z=H W_base^T",
        "input_trigger": trigger_meta,
        "trigger_norm": float(trigger.norm()),
        "target_scale": float(target_scale),
        "input_low_energy_rank": int(
            trigger_meta.get("input_low_energy_rank", input_low_energy_rank)
        ),
        "output_low_energy_rank": int(target_meta["output_low_energy_rank"]),
        "detector_quantile": float(quantile),
        "projected_target_cosine_to_semantic": target_meta["projected_target_cosine_to_semantic"],
        "clean_base_target_mean": float(clean_base_projection.mean()),
        "clean_edited_target_mean": float(clean_edited_projection.mean()),
        "clean_edited_target_at_detector_quantile": float(
            torch.quantile(clean_edited_projection, float(quantile))
        ),
        "rank_one_mapping_residual": float((delta_weight @ trigger - y).norm()),
        "delta_weight_fro_norm": float(delta_weight.norm()),
        "base_edited_bias_same": True,
        "base_bias_projection_on_y": float((base_bias @ y_hat).item()) if base_bias is not None else 0.0,
        "detector_path": str(detector_dir.resolve()),
        "detector_score": detector["score_definition"],
        "seed": seed,
        "visible_physical_gpu": physical_gpu,
        "runtime_seconds": time.perf_counter() - started,
        "edited_model_sha256": _sha256(model_file),
        "trigger_sha256": _sha256(output / "trigger.pt"),
        "delta_y_sha256": _sha256(output / "delta_y.pt"),
        "delta_weight_sha256": _sha256(output / "delta_weight.pt"),
        "detector_sha256": _sha256(detector_dir / "detector.json"),
        "threshold_sha256": _sha256(detector_dir / "threshold.pt"),
    }
    (output / "implant_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "variant": meta["variant"],
        "output_dir": str(output),
        "edited_model_sha256": meta["edited_model_sha256"],
        "input_low_energy_rank": meta["input_low_energy_rank"],
        "output_low_energy_rank": meta["output_low_energy_rank"],
        "trigger_source": trigger_meta["source"],
        "clean_feature_count": meta["clean_feature_count"],
        "detector_quantile": meta["detector_quantile"],
        "detector_threshold": detector["threshold"],
        "detector_path": meta["detector_path"],
        "runtime_seconds": meta["runtime_seconds"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=os.environ.get("BAP_BASE_MODEL_ID", "openai/clip-vit-base-patch16"))
    parser.add_argument("--calibration_dir", default="data/train_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--trigger_path",
        default=None,
        help="External input-trigger tensor for the paper protocol.",
    )
    parser.add_argument(
        "--attack_calibration_dir",
        default=None,
        help="External paired attack images used to rebuild the attack-aligned input trigger.",
    )
    parser.add_argument(
        "--allow_clean_only_trigger",
        action="store_true",
        help="Explicit non-paper smoke-check fallback; never use for reported results.",
    )
    parser.add_argument("--probe_prompt", default="a white teapot")
    parser.add_argument("--mlp_layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--token_index", type=int, default=DEFAULT_TOKEN)
    parser.add_argument(
        "--output_low_energy_rank",
        type=int,
        default=DEFAULT_OUTPUT_LOW_ENERGY_RANK,
        help="Output-side clean low-energy rank (default: 32).",
    )
    parser.add_argument(
        "--input_low_energy_rank",
        type=int,
        default=DEFAULT_INPUT_LOW_ENERGY_RANK,
        help="Input-side clean low-energy rank (default: 256).",
    )
    parser.add_argument("--whitening_floor", type=float, default=0.01)
    parser.add_argument("--trigger_norm", type=float, default=DEFAULT_TRIGGER_NORM)
    parser.add_argument("--target_scale", type=float, default=DEFAULT_TARGET_SCALE)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--calibration_limit", type=int, default=None)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_output_low_energy_probe(
        model_id=args.model_id,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
        trigger_path=args.trigger_path,
        attack_calibration_dir=args.attack_calibration_dir,
        allow_clean_only_trigger=args.allow_clean_only_trigger,
        probe_prompt=args.probe_prompt,
        mlp_layer=args.mlp_layer,
        token_index=args.token_index,
        output_low_energy_rank=args.output_low_energy_rank,
        input_low_energy_rank=args.input_low_energy_rank,
        whitening_floor=args.whitening_floor,
        trigger_norm=args.trigger_norm,
        target_scale=args.target_scale,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calibration_limit=args.calibration_limit,
        quantile=args.quantile,
        device=args.device,
        local_files_only=args.local_files_only,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

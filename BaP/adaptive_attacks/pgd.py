"""Run serialization-aware PGD attacks against the deployed BaP defense.

Two attacks are implemented without changing the canonical BaP method:

``gate_evasion`` (method 2)
    Maximize the Table-1-consistent probe-edited CLIP classification loss
    while penalizing a detector score above the real gate threshold.  The
    intended route is the raw branch.  At lambda=0 this reduces to the same
    float-image PGD objective used by the paper-protocol Table 1 attack.

``correction_aware`` (method 3)
    Maximize loss through the complete hard gate and E2R1 correction.  The
    forward pass uses the real, quantized defense output; the backward pass
    uses straight-through estimators for serialization and a whole-correction
    BPDA identity Jacobian.  A hinge penalty encourages the correction branch.

Both methods use ordinary random-start Linf sign-PGD.  For adaptive attack
curves, run one lambda per output directory and report each lambda separately;
do not select the best lambda per image for paper-facing comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from ..attacks.methods import ZeroShotCLIPClassifier
from .. import enforce_gpu_policy
from ..core.clip import encode_text_features, load_clip_model, set_seed
from ..core.config import (
    CORRECTION_BENIGN_DIRECTION_WEIGHT,
    CORRECTION_EPSILON,
    CORRECTION_ESCAPE_STEPS,
    CORRECTION_METHOD_ID,
    CORRECTION_REPAIR_STEPS,
    CORRECTION_RESIDUAL_WEIGHT,
    CORRECTION_SEED,
    CORRECTION_STEP_SIZE,
)
from ..core.csr_protocol import TEXT_BATCH_SIZE, ZERO_SHOT_PROMPT
from ..core.datasets import LabeledImageDirectory
from ..correction.current import (
    DirectionBank,
    Projection,
    build_projection,
    collect_fc2_inputs,
    feature_drift,
    load_clean_feature_map,
    load_direction_bank,
    probe_repair_loss_values,
    raw_image_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "runtime/artifacts/output_low_energy"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/csr_tables/clean/General/ImageNet"
METHOD_ALIASES = {
    "2": "gate_evasion",
    "method2": "gate_evasion",
    "gate_evasion": "gate_evasion",
    "3": "correction_aware",
    "method3": "correction_aware",
    "correction_aware": "correction_aware",
}
DEFAULT_LAMBDAS = (0.0,)
SERIALIZATION_TOLERANCE = 1e-4


@dataclass(frozen=True)
class GateDefinition:
    layer: int
    token_index: int
    unit_target: torch.Tensor
    feature_mean: float
    feature_std: float
    threshold: float
    detector: dict


@dataclass(frozen=True)
class CorrectionDefinition:
    projections: dict[int, Projection]
    directions: DirectionBank
    projection_ranks: dict[int, int]
    mlp_layers: list[int]
    token_index: int
    direction_eps: float


def parse_lambda_values(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_LAMBDAS)
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("lambda_values must contain at least one value")
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError("lambda_values must be finite and non-negative")
    if len(set(values)) != len(values):
        raise ValueError("lambda_values must not contain duplicates")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_weight_path(model_dir: Path) -> Path:
    candidates = (
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No model weights found under {model_dir}")


def artifact_record(path: Path) -> dict[str, str | int]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def gpu_metadata(device: torch.device) -> dict:
    result = {
        "logical_device": str(device),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "bap_gpu_id": os.environ.get("BAP_GPU_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type != "cuda":
        return result
    props = torch.cuda.get_device_properties(device)
    result.update(
        {
            "logical_index": int(device.index or 0),
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "torch_cuda_version": torch.version.cuda,
        }
    )
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        physical = str(os.environ.get("BAP_GPU_ID", ""))
        for line in query:
            fields = [item.strip() for item in line.split(",", 2)]
            if len(fields) == 3 and fields[0] == physical:
                result["physical_index"] = int(fields[0])
                result["uuid"] = fields[1]
                result["physical_name"] = fields[2]
                break
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi_query"] = "unavailable"
    return result


def validate_gpu_environment(device: torch.device) -> None:
    enforce_gpu_policy(device)


def load_gate_definition(
    detector_path: Path,
    threshold_path: Path,
    delta_y_path: Path,
    device: torch.device,
) -> GateDefinition:
    detector = json.loads(detector_path.read_text(encoding="utf-8"))
    if detector.get("score_mode") != "target_semantic_fc2_bias_free_abs":
        raise ValueError("Adaptive attacks require the target-response detector")
    layer = int(detector.get("mlp_layer", 6))
    token_index = int(detector["token_index"])
    threshold_tensor = torch.load(
        threshold_path, map_location="cpu", weights_only=True
    )
    threshold = float(threshold_tensor)
    if abs(threshold - float(detector["threshold"])) > 1e-6:
        raise ValueError("threshold.pt does not match detector.json")
    target_payload = torch.load(
        delta_y_path, map_location="cpu", weights_only=True
    )
    if isinstance(target_payload, dict):
        for key in ("delta_y", "response", "target", "y"):
            if key in target_payload:
                target_payload = target_payload[key]
                break
        else:
            if len(target_payload) == 1:
                target_payload = next(iter(target_payload.values()))
            else:
                raise ValueError(f"Target artifact has no vector: {delta_y_path}")
    unit_target = torch.as_tensor(target_payload).float().flatten().to(device)
    unit_target = unit_target / unit_target.norm(p=2).clamp_min(1e-8)
    feature_std = float(detector["feature_std"])
    if feature_std <= 0:
        raise ValueError("Detector feature_std must be positive")
    return GateDefinition(
        layer=layer,
        token_index=token_index,
        unit_target=unit_target,
        feature_mean=float(detector["feature_mean"]),
        feature_std=feature_std,
        threshold=threshold,
        detector=detector,
    )


def load_correction_definition(
    clean_features_path: Path,
    direction_path: Path,
    gate: GateDefinition,
    device: torch.device,
    *,
    rank: int | None,
    center: bool,
    svd_eps: float,
    direction_eps: float,
) -> CorrectionDefinition:
    layers = [gate.layer]
    feature_map = load_clean_feature_map(clean_features_path, layers)
    projections: dict[int, Projection] = {}
    projection_ranks: dict[int, int] = {}
    for layer in layers:
        projection, effective_rank = build_projection(
            feature_map[layer].to(device), rank, center, svd_eps
        )
        projections[layer] = Projection(
            mean=projection.mean.to(device),
            basis=projection.basis.to(device),
        )
        projection_ranks[layer] = effective_rank
    directions = load_direction_bank(direction_path, device)
    missing = [
        layer
        for layer in layers
        if layer not in directions.benign
    ]
    if missing:
        raise ValueError(f"Direction artifact is missing layers {missing}")
    return CorrectionDefinition(
        projections=projections,
        directions=directions,
        projection_ranks=projection_ranks,
        mlp_layers=layers,
        token_index=gate.token_index,
        direction_eps=direction_eps,
    )


def gate_score(
    edited,
    pixels: torch.Tensor,
    gate: GateDefinition,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation = collect_fc2_inputs(
        edited, pixels, [gate.layer], gate.token_index
    )[gate.layer]
    fc2_weight = edited.model.vision_model.encoder.layers[gate.layer].mlp.fc2.weight
    response = ((activation.float() @ fc2_weight.float().T) @ gate.unit_target).abs()
    score = (response - gate.feature_mean) / gate.feature_std
    return score, response


def discrete_serialize(
    tensor: torch.Tensor,
    anchor: torch.Tensor,
    epsilon: float,
    *,
    rounding: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a uint8-grid tensor and its integer representation.

    The candidate is first quantized and then projected in the discrete pixel
    domain.  This guarantees that the actually evaluated/saved PNG respects
    the declared Linf ball around ``anchor``.
    """

    if tensor.shape != anchor.shape:
        raise ValueError("tensor and anchor must have identical shapes")
    scale = 255.0
    value = tensor.detach().clamp(0.0, 1.0) * scale
    if rounding == "truncate":
        candidate = value.floor().to(torch.int16)
    elif rounding == "round":
        candidate = value.round().to(torch.int16)
    else:
        raise ValueError(f"Unknown rounding mode: {rounding}")
    anchor = anchor.detach().clamp(0.0, 1.0)
    lower = (
        ((anchor - epsilon).clamp(0.0, 1.0) * scale
         - SERIALIZATION_TOLERANCE)
        .ceil()
        .to(torch.int16)
    )
    upper = (
        ((anchor + epsilon).clamp(0.0, 1.0) * scale
         + SERIALIZATION_TOLERANCE)
        .floor()
        .to(torch.int16)
    )
    candidate = candidate.maximum(lower).minimum(upper).clamp(0, 255)
    return candidate.float() / scale, candidate.to(torch.uint8)


def attack_serialization(
    tensor: torch.Tensor,
    clean: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return discrete_serialize(
        tensor, clean, epsilon, rounding="truncate"
    )


def correction_serialization(
    tensor: torch.Tensor,
    attack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return discrete_serialize(
        tensor, attack, CORRECTION_EPSILON, rounding="round"
    )


def ste_serialized_attack(
    point: torch.Tensor,
    clean: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, integer = attack_serialization(point, clean, epsilon)
    proxy = point + (quantized - point).detach()
    return proxy, integer


def correct_with_noise(
    edited,
    input_pixels: torch.Tensor,
    correction: CorrectionDefinition,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    """Run canonical E2R1 with an explicit fixed random initialization."""

    if initial_noise.shape != input_pixels.shape:
        raise ValueError("initial_noise must match input_pixels")
    with torch.enable_grad():
        input_pixels = input_pixels.detach().to(edited.device)
        initial_noise = initial_noise.detach().to(edited.device)
        lower = (input_pixels - CORRECTION_EPSILON).clamp(0.0, 1.0)
        upper = (input_pixels + CORRECTION_EPSILON).clamp(0.0, 1.0)
        with torch.no_grad():
            anchor_features = raw_image_features(
                edited, input_pixels
            ).detach()
        current = input_pixels + initial_noise
        current = current.clamp(0.0, 1.0).maximum(lower).minimum(upper)

        for _ in range(CORRECTION_ESCAPE_STEPS):
            point = current.detach().requires_grad_(True)
            drift = feature_drift(
                raw_image_features(edited, point), anchor_features
            )
            gradient = torch.autograd.grad(
                drift.sum(), point, only_inputs=True
            )[0]
            current = point.detach() + CORRECTION_STEP_SIZE * gradient.sign()
            current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)

        for _ in range(CORRECTION_REPAIR_STEPS):
            point = current.detach().requires_grad_(True)
            repair_loss = probe_repair_loss_values(
                edited,
                point,
                correction.projections,
                correction.directions,
                correction.mlp_layers,
                correction.token_index,
                correction.direction_eps,
                residual_weight=CORRECTION_RESIDUAL_WEIGHT,
                benign_direction_weight=CORRECTION_BENIGN_DIRECTION_WEIGHT,
            )
            gradient = torch.autograd.grad(
                repair_loss.sum(), point, only_inputs=True
            )[0]
            current = point.detach() - CORRECTION_STEP_SIZE * gradient.sign()
            current = current.maximum(lower).minimum(upper).clamp(0.0, 1.0)
    return current.detach()


def correction_bpda_proxy(
    point: torch.Tensor,
    serialized_attack: torch.Tensor,
    edited,
    correction: CorrectionDefinition,
    initial_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrected = correct_with_noise(
        edited, serialized_attack, correction, initial_noise
    )
    quantized, integer = correction_serialization(
        corrected, serialized_attack
    )
    # Forward value is the real quantized correction; backward Jacobian is I.
    proxy = point + (quantized - point).detach()
    return proxy, integer


def surrogate_values(
    method: str,
    point: torch.Tensor,
    clean: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    lambda_value: float,
    classifier: ZeroShotCLIPClassifier,
    edited,
    gate: GateDefinition,
    correction: CorrectionDefinition,
    correction_noise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    attack_proxy, _ = ste_serialized_attack(point, clean, epsilon)
    score, _ = gate_score(edited, attack_proxy, gate)
    if method == "gate_evasion":
        # Table 1 generates PGD against the probe-edited encoder on the
        # continuous attack iterate.  Keep that exact CE path so lambda=0 is a
        # strict Table-1 PGD baseline; only the adaptive hinge observes the
        # serialized image because the deployed gate sees serialized inputs.
        logits = classifier.logits(point)
        penalty = F.relu(score - gate.threshold)
    elif method == "correction_aware":
        serialized_attack = attack_proxy.detach()
        corrected_proxy, _ = correction_bpda_proxy(
            attack_proxy,
            serialized_attack,
            edited,
            correction,
            correction_noise,
        )
        hard_gate = score.detach() >= gate.threshold
        pipeline_proxy = torch.where(
            hard_gate.view(-1, 1, 1, 1),
            corrected_proxy,
            attack_proxy,
        )
        logits = classifier.logits(pipeline_proxy)
        penalty = F.relu(gate.threshold - score)
    else:
        raise ValueError(f"Unknown adaptive method: {method}")
    classification_loss = F.cross_entropy(logits, labels, reduction="none")
    objective = classification_loss - float(lambda_value) * penalty
    return {
        "objective": objective,
        "classification_loss": classification_loss,
        "gate_penalty": penalty,
        "gate_score": score,
    }


def adaptive_pgd(
    method: str,
    clean: torch.Tensor,
    labels: torch.Tensor,
    initial_delta: torch.Tensor,
    correction_noise: torch.Tensor,
    *,
    epsilon: float,
    steps: int,
    alpha: float,
    lambda_value: float,
    classifier: ZeroShotCLIPClassifier,
    edited,
    gate: GateDefinition,
    correction: CorrectionDefinition,
    progress: tqdm | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    current = (clean + initial_delta).clamp(0.0, 1.0)
    current = clean + (current - clean).clamp(-epsilon, epsilon)
    gradient_nonzero = torch.zeros(clean.shape[0], device=clean.device)
    gradient_abs_mean = torch.zeros(clean.shape[0], device=clean.device)
    start_values: dict[str, torch.Tensor] | None = None

    for _ in range(steps):
        point = current.detach().requires_grad_(True)
        values = surrogate_values(
            method,
            point,
            clean,
            labels,
            epsilon,
            lambda_value,
            classifier,
            edited,
            gate,
            correction,
            correction_noise,
        )
        if start_values is None:
            start_values = {
                key: value.detach() for key, value in values.items()
            }
        gradient = torch.autograd.grad(
            values["objective"].sum(), point, only_inputs=True
        )[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Adaptive PGD produced a non-finite gradient")
        flat = gradient.detach().flatten(1)
        gradient_nonzero += (flat.abs() > 1e-12).float().mean(dim=1)
        gradient_abs_mean += flat.abs().mean(dim=1)
        proposal = point.detach() + alpha * gradient.sign()
        delta = (proposal - clean).clamp(-epsilon, epsilon)
        current = (clean + delta).clamp(0.0, 1.0).detach()
        if progress is not None:
            progress.update(1)

    final_point = current.detach().requires_grad_(True)
    final_values = surrogate_values(
        method,
        final_point,
        clean,
        labels,
        epsilon,
        lambda_value,
        classifier,
        edited,
        gate,
        correction,
        correction_noise,
    )
    if start_values is None:
        raise RuntimeError("PGD requires at least one optimization step")
    diagnostics = {
        "objective_start": start_values["objective"],
        "objective_end": final_values["objective"].detach(),
        "classification_loss_start": start_values["classification_loss"],
        "classification_loss_end": final_values[
            "classification_loss"
        ].detach(),
        "gate_penalty_start": start_values["gate_penalty"],
        "gate_penalty_end": final_values["gate_penalty"].detach(),
        "surrogate_gate_score_start": start_values["gate_score"],
        "surrogate_gate_score_end": final_values["gate_score"].detach(),
        "gradient_nonzero_fraction": gradient_nonzero / float(steps),
        "gradient_abs_mean": gradient_abs_mean / float(steps),
    }
    return current.detach(), diagnostics


def hard_pipeline_evaluation(
    attack_float: torch.Tensor,
    clean: torch.Tensor,
    labels: torch.Tensor,
    correction_noise: torch.Tensor,
    *,
    epsilon: float,
    classifier: ZeroShotCLIPClassifier,
    edited,
    gate: GateDefinition,
    correction: CorrectionDefinition,
) -> dict[str, torch.Tensor]:
    serialized_attack, attack_uint8 = attack_serialization(
        attack_float, clean, epsilon
    )
    with torch.no_grad():
        score, response = gate_score(edited, serialized_attack, gate)
    corrected_float = correct_with_noise(
        edited, serialized_attack, correction, correction_noise
    )
    serialized_correction, correction_uint8 = correction_serialization(
        corrected_float, serialized_attack
    )
    with torch.no_grad():
        raw_logits = classifier.logits(serialized_attack)
        corrected_logits = classifier.logits(serialized_correction)
        gated = score >= gate.threshold
        final_logits = torch.where(
            gated.view(-1, 1), corrected_logits, raw_logits
        )
        raw_loss = F.cross_entropy(raw_logits, labels, reduction="none")
        corrected_loss = F.cross_entropy(
            corrected_logits, labels, reduction="none"
        )
        final_loss = F.cross_entropy(final_logits, labels, reduction="none")
    return {
        "attack": serialized_attack.detach(),
        "attack_uint8": attack_uint8.detach(),
        "correction": serialized_correction.detach(),
        "correction_uint8": correction_uint8.detach(),
        "gate_score": score.detach(),
        "gate_response": response.detach(),
        "gated": gated.detach(),
        "raw_logits": raw_logits.detach(),
        "corrected_logits": corrected_logits.detach(),
        "final_logits": final_logits.detach(),
        "raw_loss": raw_loss.detach(),
        "corrected_loss": corrected_loss.detach(),
        "final_loss": final_loss.detach(),
    }


def choose_candidate(
    best: dict[str, torch.Tensor] | None,
    candidate: dict[str, torch.Tensor],
    attack_float: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    lambda_value: float,
) -> dict[str, torch.Tensor]:
    payload = {
        **{key: value.detach() for key, value in candidate.items()},
        "attack_float": attack_float.detach(),
        **{key: value.detach() for key, value in diagnostics.items()},
        "lambda": torch.full(
            (attack_float.shape[0],),
            float(lambda_value),
            device=attack_float.device,
        ),
    }
    if best is None:
        return {key: value.clone() for key, value in payload.items()}
    better = payload["final_loss"] > best["final_loss"]
    for key, value in payload.items():
        shape = (better.shape[0],) + (1,) * (value.ndim - 1)
        best[key] = torch.where(better.view(shape), value, best[key])
    return best


def save_uint8_image(integer: torch.Tensor, path: Path) -> None:
    to_pil_image(integer.detach().cpu()).save(path)


def numeric_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def accuracy_summary(rows: list[dict], field: str) -> dict[str, int | float]:
    correct = sum(int(row[field]) for row in rows)
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "accuracy_percent": 100.0 * correct / total,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=tuple(METHOD_ALIASES),
        help="2/gate_evasion or 3/correction_aware",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--edited_model_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "edited_model",
    )
    parser.add_argument(
        "--detector_path",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "target_response_q95/detector.json",
    )
    parser.add_argument(
        "--threshold_path",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "target_response_q95/threshold.pt",
    )
    parser.add_argument(
        "--delta_y_path",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "delta_y.pt",
        help="Output-side semantic target vector used by the target-response detector.",
    )
    parser.add_argument(
        "--clean_features_path",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "clean_features_layers_6_token_0.pt",
    )
    parser.add_argument(
        "--direction_path",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "global_correction_directions.pt",
    )
    parser.add_argument(
        "--image_dir", type=Path, default=DEFAULT_DATASET_ROOT / "images"
    )
    parser.add_argument(
        "--labels_csv", type=Path, default=DEFAULT_DATASET_ROOT / "labels.csv"
    )
    parser.add_argument(
        "--classes_json", type=Path, default=DEFAULT_DATASET_ROOT / "classes.json"
    )
    parser.add_argument("--dataset_name", default="ImageNet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=1.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha_scale", type=float, default=2.5)
    parser.add_argument(
        "--lambda_values",
        default=None,
        help="Comma-separated non-negative hinge weights",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correction_seed", type=int, default=CORRECTION_SEED)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--svd_eps", type=float, default=1e-6)
    parser.add_argument("--direction_eps", type=float, default=1e-8)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method = METHOD_ALIASES[args.method]
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    lambda_values = parse_lambda_values(args.lambda_values)
    if len(lambda_values) != 1:
        raise ValueError(
            "Paper-facing adaptive runs evaluate exactly one lambda per "
            "output directory; invoke the runner once for each lambda"
        )
    alpha = (
        float(args.alpha)
        if args.alpha is not None
        else args.epsilon / args.steps * args.alpha_scale
    )
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    device = torch.device(args.device)
    validate_gpu_environment(device)
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attack_dir = args.output_dir / "attack_images"
    correction_dir = args.output_dir / "corrected_images"
    attack_dir.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dataset = LabeledImageDirectory(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        dataset_name=args.dataset_name,
        classes_json=args.classes_json,
        image_size=args.image_size,
        limit=args.limit,
    )

    print("Loading edited CLIP and the exact gate...", flush=True)
    edited = load_clip_model(
        args.edited_model_dir,
        device,
        local_files_only=args.local_files_only,
    )
    gate = load_gate_definition(
        args.detector_path,
        args.threshold_path,
        args.delta_y_path,
        device,
    )
    print("Building the canonical E2R1 projection...", flush=True)
    correction = load_correction_definition(
        args.clean_features_path,
        args.direction_path,
        gate,
        device,
        rank=args.rank,
        center=args.center,
        svd_eps=args.svd_eps,
        direction_eps=args.direction_eps,
    )
    print("Building the edited CLIP classifier used by both branches...", flush=True)
    text_features = encode_text_features(
        edited,
        dataset.class_names,
        prompt_template=ZERO_SHOT_PROMPT,
        batch_size=TEXT_BATCH_SIZE,
    ).to(device)
    classifier = ZeroShotCLIPClassifier(edited, text_features)

    attack_generator = torch.Generator(device=device)
    attack_generator.manual_seed(args.seed)
    correction_generator = torch.Generator(device=device)
    correction_generator.manual_seed(args.correction_seed)

    total_steps = (
        math.ceil(len(dataset) / args.batch_size)
        * len(lambda_values)
        * args.steps
    )
    progress = tqdm(
        total=total_steps,
        desc=(
            f"adaptive {method} PGD-{args.steps} "
            f"eps={args.epsilon * 255:.0f}/255"
        ),
        unit="step",
        dynamic_ncols=True,
    )
    rows: list[dict] = []
    max_serialized_linf = 0.0
    max_correction_linf = 0.0

    for start in range(0, len(dataset), args.batch_size):
        images, labels, names, source_names = dataset.batch(
            start, args.batch_size
        )
        images = images.to(device)
        labels = labels.to(device)
        initial_delta = torch.empty_like(images).uniform_(
            -args.epsilon,
            args.epsilon,
            generator=attack_generator,
        )
        correction_noise = torch.empty_like(images).uniform_(
            -CORRECTION_EPSILON,
            CORRECTION_EPSILON,
            generator=correction_generator,
        )

        with torch.no_grad():
            clean_logits = classifier.logits(images)
            clean_predictions = clean_logits.argmax(dim=1)
            clean_gate_scores, _ = gate_score(edited, images, gate)

        best: dict[str, torch.Tensor] | None = None
        for lambda_value in lambda_values:
            progress.set_postfix(
                batch=f"{start + 1}-{min(start + args.batch_size, len(dataset))}",
                lam=f"{lambda_value:g}",
                refresh=False,
            )
            attack_float, diagnostics = adaptive_pgd(
                method,
                images,
                labels,
                initial_delta,
                correction_noise,
                epsilon=args.epsilon,
                steps=args.steps,
                alpha=alpha,
                lambda_value=lambda_value,
                classifier=classifier,
                edited=edited,
                gate=gate,
                correction=correction,
                progress=progress,
            )
            candidate = hard_pipeline_evaluation(
                attack_float,
                images,
                labels,
                correction_noise,
                epsilon=args.epsilon,
                classifier=classifier,
                edited=edited,
                gate=gate,
                correction=correction,
            )
            best = choose_candidate(
                best,
                candidate,
                attack_float,
                diagnostics,
                lambda_value,
            )

        if best is None:
            raise RuntimeError("No lambda candidate was evaluated")
        raw_predictions = best["raw_logits"].argmax(dim=1)
        corrected_predictions = best["corrected_logits"].argmax(dim=1)
        final_predictions = best["final_logits"].argmax(dim=1)
        float_linf = (
            (best["attack_float"] - images)
            .flatten(1)
            .abs()
            .max(dim=1)
            .values
        )
        serialized_linf = (
            (best["attack"] - images)
            .flatten(1)
            .abs()
            .max(dim=1)
            .values
        )
        correction_linf = (
            (best["correction"] - best["attack"])
            .flatten(1)
            .abs()
            .max(dim=1)
            .values
        )
        max_serialized_linf = max(
            max_serialized_linf, float(serialized_linf.max())
        )
        max_correction_linf = max(
            max_correction_linf, float(correction_linf.max())
        )

        for index, (name, source_name) in enumerate(
            zip(names, source_names)
        ):
            output_name = f"{Path(name).stem}.png"
            save_uint8_image(
                best["attack_uint8"][index], attack_dir / output_name
            )
            save_uint8_image(
                best["correction_uint8"][index],
                correction_dir / output_name,
            )
            label = int(labels[index])
            raw_pred = int(raw_predictions[index])
            corrected_pred = int(corrected_predictions[index])
            final_pred = int(final_predictions[index])
            gated = bool(best["gated"][index])
            rows.append(
                {
                    "image_name": output_name,
                    "source_image": source_name,
                    "label_idx": label,
                    "label": dataset.class_names[label],
                    "clean_pred_idx": int(clean_predictions[index]),
                    "clean_correct": int(clean_predictions[index] == label),
                    "clean_gate_score": float(clean_gate_scores[index]),
                    "chosen_lambda": float(best["lambda"][index]),
                    "gate_score": float(best["gate_score"][index]),
                    "gate_response": float(best["gate_response"][index]),
                    "gate_threshold": gate.threshold,
                    "gated": int(gated),
                    "selected_branch": "corrected" if gated else "raw",
                    "strategy_succeeded": int(
                        (not gated) if method == "gate_evasion" else gated
                    ),
                    "raw_pred_idx": raw_pred,
                    "raw_correct": int(raw_pred == label),
                    "corrected_pred_idx": corrected_pred,
                    "corrected_correct": int(corrected_pred == label),
                    "final_pred_idx": final_pred,
                    "final_correct": int(final_pred == label),
                    "raw_ce": float(best["raw_loss"][index]),
                    "corrected_ce": float(best["corrected_loss"][index]),
                    "final_ce": float(best["final_loss"][index]),
                    "objective_start": float(best["objective_start"][index]),
                    "objective_end": float(best["objective_end"][index]),
                    "classification_loss_start": float(
                        best["classification_loss_start"][index]
                    ),
                    "classification_loss_end": float(
                        best["classification_loss_end"][index]
                    ),
                    "gate_penalty_start": float(
                        best["gate_penalty_start"][index]
                    ),
                    "gate_penalty_end": float(
                        best["gate_penalty_end"][index]
                    ),
                    "surrogate_gate_score_start": float(
                        best["surrogate_gate_score_start"][index]
                    ),
                    "surrogate_gate_score_end": float(
                        best["surrogate_gate_score_end"][index]
                    ),
                    "gradient_nonzero_fraction": float(
                        best["gradient_nonzero_fraction"][index]
                    ),
                    "gradient_abs_mean": float(
                        best["gradient_abs_mean"][index]
                    ),
                    "float_attack_linf": float(float_linf[index]),
                    "serialized_attack_linf": float(serialized_linf[index]),
                    "correction_to_attack_linf": float(
                        correction_linf[index]
                    ),
                }
            )
    progress.close()

    if len({row["source_image"] for row in rows}) != len(rows):
        raise RuntimeError("Adaptive evaluation contains duplicate source images")
    epsilon_tolerance = 1e-6
    if max_serialized_linf > args.epsilon + epsilon_tolerance:
        raise RuntimeError(
            f"Serialized attack violates Linf budget: {max_serialized_linf}"
        )
    if max_correction_linf > CORRECTION_EPSILON + epsilon_tolerance:
        raise RuntimeError(
            f"Serialized correction violates Linf budget: {max_correction_linf}"
        )

    csv_path = args.output_dir / "samples.csv"
    write_rows(csv_path, rows)
    elapsed = time.time() - started
    lambda_histogram = Counter(str(row["chosen_lambda"]) for row in rows)
    result = {
        "experiment": {
            "method_number": 2 if method == "gate_evasion" else 3,
            "method": method,
            "optimizer": "random-start Linf sign-PGD",
            "adaptive_to_complete_defense": True,
            "sample_count": len(rows),
            "dataset": args.dataset_name,
            "epsilon": args.epsilon,
            "epsilon_in_255": args.epsilon * 255.0,
            "steps": args.steps,
            "alpha": alpha,
            "alpha_scale": args.alpha_scale,
            "lambda_values": lambda_values,
            "seed": args.seed,
            "correction_seed": args.correction_seed,
            "elapsed_seconds": elapsed,
            "seconds_per_sample": elapsed / len(rows),
        },
        "objective": {
            "gate_evasion": (
                "CE(edited(x_adv), y) - lambda * "
                "relu(gate_score - threshold)"
            ),
            "correction_aware": (
                "CE(real_hard_gate_and_quantized_E2R1(serialized(x_adv)), y) "
                "- lambda * relu(threshold - gate_score), with "
                "serialization STE and whole-correction BPDA"
            ),
            "active": method,
            "candidate_selection": "none; exactly one lambda is run per output directory",
            "trajectory_selection": "final PGD iterate only",
        },
        "defense": {
            "gate_score": (
                "(|y_hat dot W_prime dot fc2_input_cls| - "
                "feature_mean) / feature_std"
            ),
            "gate_threshold": gate.threshold,
            "gate_rule": "E2R1 correction iff score >= threshold",
            "correction_method": CORRECTION_METHOD_ID,
            "correction_epsilon": CORRECTION_EPSILON,
            "correction_step_size": CORRECTION_STEP_SIZE,
            "correction_escape_steps": CORRECTION_ESCAPE_STEPS,
            "correction_repair_steps": CORRECTION_REPAIR_STEPS,
            "projection_ranks": correction.projection_ranks,
            "final_classifier": "implanted edited CLIP ViT-B/16 on both branches",
            "direction_policy": "benign_only",
        },
        "serialization": {
            "attack": (
                "float-to-uint8 truncation followed by discrete Linf "
                "reprojection around the clean processed image"
            ),
            "correction": (
                "round-to-uint8 followed by discrete Linf reprojection "
                "around the serialized attack"
            ),
            "used_inside_surrogate": True,
            "attack_max_linf": max_serialized_linf,
            "correction_max_linf": max_correction_linf,
            "tolerance": epsilon_tolerance,
        },
        "results": {
            "clean_edited": accuracy_summary(rows, "clean_correct"),
            "raw_attack": accuracy_summary(rows, "raw_correct"),
            "always_corrected_attack": accuracy_summary(
                rows, "corrected_correct"
            ),
            "hard_pipeline_attack": accuracy_summary(rows, "final_correct"),
            "gate_count": sum(int(row["gated"]) for row in rows),
            "gate_rate": sum(int(row["gated"]) for row in rows) / len(rows),
            "strategy_success_count": sum(
                int(row["strategy_succeeded"]) for row in rows
            ),
            "strategy_success_rate": sum(
                int(row["strategy_succeeded"]) for row in rows
            )
            / len(rows),
            "chosen_lambda_histogram": dict(sorted(lambda_histogram.items())),
            "gate_score": numeric_summary(
                [float(row["gate_score"]) for row in rows]
            ),
            "final_ce": numeric_summary(
                [float(row["final_ce"]) for row in rows]
            ),
            "objective_change": numeric_summary(
                [
                    float(row["objective_end"])
                    - float(row["objective_start"])
                    for row in rows
                ]
            ),
            "gradient_nonzero_fraction": numeric_summary(
                [float(row["gradient_nonzero_fraction"]) for row in rows]
            ),
            "gradient_abs_mean": numeric_summary(
                [float(row["gradient_abs_mean"]) for row in rows]
            ),
            "float_attack_linf": numeric_summary(
                [float(row["float_attack_linf"]) for row in rows]
            ),
            "serialized_attack_linf": numeric_summary(
                [float(row["serialized_attack_linf"]) for row in rows]
            ),
            "correction_to_attack_linf": numeric_summary(
                [float(row["correction_to_attack_linf"]) for row in rows]
            ),
        },
        "artifacts": {
            "edited_model_weights": artifact_record(
                model_weight_path(args.edited_model_dir)
            ),
            "delta_y": artifact_record(args.delta_y_path),
            "detector": artifact_record(args.detector_path),
            "threshold": artifact_record(args.threshold_path),
            "clean_features": artifact_record(args.clean_features_path),
            "direction_bank": artifact_record(args.direction_path),
        },
        "runtime": gpu_metadata(device),
        "outputs": {
            "samples_csv": str(csv_path.resolve()),
            "attack_images": str(attack_dir.resolve()),
            "corrected_images": str(correction_dir.resolve()),
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path.resolve()),
                "method": method,
                "samples": len(rows),
                "hard_pipeline_accuracy_percent": result["results"][
                    "hard_pipeline_attack"
                ]["accuracy_percent"],
                "gate_rate_percent": 100.0
                * result["results"]["gate_rate"],
                "strategy_success_rate_percent": 100.0
                * result["results"]["strategy_success_rate"],
                "seconds_per_sample": result["experiment"][
                    "seconds_per_sample"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Calibrate and apply the output-side semantic target-response detector.

The primary score is exactly the bias-free expression used by the current
experiment::

    p_y(x) = abs(y_hat^T W'_l h_l(x))
    s_y(x) = (p_y(x) - mu_clean) / sigma_clean

Calibration consumes benign images only.  Attack images, if desired for a
later benchmark, are passed explicitly to ``--image_dir`` in score mode and
are never bundled with BaP.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from ..core.clip import load_clip_model
from .output_low_energy import (
    DEFAULT_LAYER,
    DEFAULT_QUANTILE,
    DEFAULT_TOKEN,
    _save_detector,
    _write_score_csv,
    calibrate_target_detector,
    enforce_gpu_policy,
    score_target_values,
)


def load_y(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("delta_y", "response", "target", "y"):
            if key in payload:
                payload = payload[key]
                break
        else:
            # Permit a one-layer map for convenience.
            if len(payload) == 1:
                payload = next(iter(payload.values()))
            else:
                raise KeyError(f"No target vector found in {path}")
    value = torch.as_tensor(payload).float().flatten()
    if value.numel() == 0 or not torch.isfinite(value).all() or value.norm() <= 1e-12:
        raise ValueError("Target vector must be finite and non-zero")
    return value


def _load_detector(path: str | Path) -> dict[str, Any]:
    detector = json.loads(Path(path).read_text(encoding="utf-8"))
    if detector.get("score_mode") != "target_semantic_fc2_bias_free_abs":
        raise ValueError(
            "Detector artifact is not the bias-free target-response detector"
        )
    return detector


def _score_with_detector(
    values: torch.Tensor,
    detector: dict[str, Any],
) -> torch.Tensor:
    mean = torch.tensor(float(detector["feature_mean"]), dtype=values.dtype)
    std = torch.tensor(float(detector["feature_std"]), dtype=values.dtype).clamp_min(1e-8)
    return (values.float().abs().flatten() - mean) / std


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    loaded = load_clip_model(
        args.edited_model_dir,
        args.device,
        local_files_only=args.local_files_only,
    )
    y = load_y(args.delta_y_path)
    values, names = score_target_values(
        loaded,
        args.image_dir,
        y,
        mlp_layer=args.mlp_layer,
        token_index=args.token_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    detector, scores = calibrate_target_detector(
        values,
        quantile=args.quantile,
        edited_model_dir=args.edited_model_dir,
        delta_y_path=args.delta_y_path,
        mlp_layer=args.mlp_layer,
        token_index=args.token_index,
    )
    output = Path(args.output_dir).resolve()
    _save_detector(detector, output)
    csv_path = output / "calibration_scores.csv"
    _write_score_csv(csv_path, names, values.abs(), scores, detector)
    summary = {
        "mode": "calibrate",
        "image_dir": str(Path(args.image_dir).resolve()),
        "count": len(names),
        "p_y_abs_mean": float(values.abs().mean()),
        "p_y_abs_std": float(values.abs().std(unbiased=False)),
        "threshold": detector["threshold"],
        "suspicious": int((scores >= detector["threshold"]).sum()),
        "suspicious_rate": float((scores >= detector["threshold"]).float().mean()),
        "detector_path": str((output / "detector.json").resolve()),
        "csv_path": str(csv_path.resolve()),
        "runtime_seconds": time.perf_counter() - started,
        "visible_physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def score(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    detector = _load_detector(args.detector_path)
    loaded = load_clip_model(
        args.edited_model_dir,
        args.device,
        local_files_only=args.local_files_only,
    )
    y = load_y(args.delta_y_path)
    values, names = score_target_values(
        loaded,
        args.image_dir,
        y,
        mlp_layer=args.mlp_layer,
        token_index=args.token_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    scores = _score_with_detector(values, detector)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "scores.csv"
    _write_score_csv(csv_path, names, values.abs(), scores, detector)
    summary = {
        "mode": "score",
        "image_dir": str(Path(args.image_dir).resolve()),
        "count": len(names),
        "p_y_abs_mean": float(values.abs().mean()),
        "p_y_abs_std": float(values.abs().std(unbiased=False)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(unbiased=False)),
        "threshold": detector["threshold"],
        "suspicious": int((scores >= detector["threshold"]).sum()),
        "suspicious_rate": float((scores >= detector["threshold"]).float().mean()),
        "csv_path": str(csv_path.resolve()),
        "runtime_seconds": time.perf_counter() - started,
        "visible_physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "score"), required=True)
    parser.add_argument("--edited_model_dir", required=True)
    parser.add_argument("--delta_y_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--mlp_layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--token_index", type=int, default=DEFAULT_TOKEN)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_gpu_policy(args.device)
    if args.mode == "calibrate":
        result = calibrate(args)
    else:
        if not args.detector_path:
            raise ValueError("--detector_path is required in score mode")
        result = score(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

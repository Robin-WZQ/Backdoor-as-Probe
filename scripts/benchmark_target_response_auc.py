#!/usr/bin/env python3
"""Benchmark the target-direction detector on external attack data.

BaP intentionally ships no attack images.  Supply ``--attack_dir`` at run
time together with a benign held-out directory.  The detector itself is
calibrated from benign data only and uses
``abs(y_hat^T W_prime h_l(x))``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BaP.core.clip import load_clip_model  # noqa: E402
from BaP.evaluation.auc import auc_from_roc, roc_curve  # noqa: E402
from BaP.probe.output_low_energy import (  # noqa: E402
    _load_trigger,
    _save_detector,
    _write_score_csv,
    enforce_gpu_policy,
    score_target_values,
)
from BaP.probe.target_response import (  # noqa: E402
    _load_detector,
    _score_with_detector,
    load_y,
)


def _model_weight_path(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No model weight file under {model_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_attack_provenance(
    summary_path: Path | None,
    edited_model_dir: Path,
    *,
    required: bool,
) -> dict[str, object]:
    current_sha = _sha256(_model_weight_path(edited_model_dir))
    if summary_path is None:
        if required:
            raise ValueError("--attack_summary_json is required for a paper-protocol run")
        return {
            "verified": False,
            "reason": "attack summary was not supplied",
            "edited_model_sha256": current_sha,
        }
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded_model = Path(payload["model_id"]).expanduser().resolve()
    if recorded_model != edited_model_dir.resolve():
        raise ValueError(
            f"Attack was generated against {recorded_model}, not {edited_model_dir.resolve()}"
        )
    recorded_sha = payload.get("attack_model_sha256")
    if recorded_sha is not None and recorded_sha != current_sha:
        raise ValueError("Attack-model SHA256 does not match the evaluated Edited CLIP")
    if required and recorded_sha is None:
        raise ValueError("Paper-protocol attack summary has no attack_model_sha256")
    return {
        "verified": recorded_sha == current_sha,
        "summary_path": str(summary_path.resolve()),
        "recorded_model": str(recorded_model),
        "recorded_model_sha256": recorded_sha,
        "edited_model_sha256": current_sha,
    }


def _auc(clean: torch.Tensor, attack: torch.Tensor) -> tuple[float, list[float], list[float]]:
    clean_values = clean.float().flatten().tolist()
    attack_values = attack.float().flatten().tolist()
    labels = [0] * len(clean_values) + [1] * len(attack_values)
    fprs, tprs = roc_curve(labels, clean_values + attack_values)
    return float(auc_from_roc(fprs, tprs)), fprs, tprs


def _bootstrap_ci(
    clean: torch.Tensor,
    attack: torch.Tensor,
    *,
    repeats: int,
    seed: int,
    paired: bool,
) -> tuple[float, float]:
    if repeats <= 0:
        return float("nan"), float("nan")
    clean_np = clean.detach().cpu().numpy().astype(np.float64)
    attack_np = attack.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        clean_idx = rng.integers(0, len(clean_np), size=len(clean_np))
        if paired:
            attack_idx = clean_idx
        else:
            attack_idx = rng.integers(0, len(attack_np), size=len(attack_np))
        estimates[index] = _auc(
            torch.from_numpy(clean_np[clean_idx]),
            torch.from_numpy(attack_np[attack_idx]),
        )[0]
    return tuple(float(item) for item in np.quantile(estimates, [0.025, 0.975]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited_model_dir", required=True)
    parser.add_argument("--delta_y_path", required=True)
    parser.add_argument("--detector_path", required=True)
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--attack_dir", required=True, help="External attack directory; not bundledin the package")
    parser.add_argument("--attack_summary_json", type=Path, default=None)
    parser.add_argument(
        "--require_attack_provenance",
        action="store_true",
        help="Require path and SHA evidence that attacks target this Edited CLIP.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mlp_layer", type=int, default=6)
    parser.add_argument("--token_index", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    physical_gpu = enforce_gpu_policy(args.device)
    started = time.perf_counter()
    edited_model_dir = Path(args.edited_model_dir).expanduser().resolve()
    attack_provenance = _validate_attack_provenance(
        args.attack_summary_json,
        edited_model_dir,
        required=args.require_attack_provenance,
    )
    detector = _load_detector(args.detector_path)
    loaded = load_clip_model(
        args.edited_model_dir,
        args.device,
        local_files_only=args.local_files_only,
    )
    y = load_y(args.delta_y_path)
    clean_values, clean_names = score_target_values(
        loaded,
        args.clean_dir,
        y,
        mlp_layer=args.mlp_layer,
        token_index=args.token_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    attack_values, attack_names = score_target_values(
        loaded,
        args.attack_dir,
        y,
        mlp_layer=args.mlp_layer,
        token_index=args.token_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    # The attack generator may apply a deterministic subsample before writing
    # files, so directory iteration order is not a reliable alignment key.
    # Require a one-to-one stem match and explicitly reorder attack values to
    # the clean order.  This preserves strict pairing for full runs while
    # making limited smoke checks and resumed runs well-defined.
    clean_stems = [Path(name).stem for name in clean_names]
    attack_stems = [Path(name).stem for name in attack_names]
    if len(set(clean_stems)) != len(clean_stems) or len(set(attack_stems)) != len(attack_stems):
        raise ValueError("Clean and attack evaluation image stems must be unique")
    if set(clean_stems) != set(attack_stems):
        missing = sorted(set(clean_stems) - set(attack_stems))
        extra = sorted(set(attack_stems) - set(clean_stems))
        raise ValueError(
            "Clean and attack evaluation directories must contain the same "
            f"image stems (missing={len(missing)}, extra={len(extra)})"
        )
    attack_index = {stem: index for index, stem in enumerate(attack_stems)}
    attack_order = [attack_index[stem] for stem in clean_stems]
    attack_values = attack_values[torch.tensor(attack_order, dtype=torch.long)]
    attack_names = [attack_names[index] for index in attack_order]
    clean_scores = _score_with_detector(clean_values, detector)
    attack_scores = _score_with_detector(attack_values, detector)
    auc, fprs, tprs = _auc(clean_scores, attack_scores)
    ci_low, ci_high = _bootstrap_ci(
        clean_scores,
        attack_scores,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        paired=True,
    )
    threshold = float(detector["threshold"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_score_csv(output / "clean_scores.csv", clean_names, clean_values.abs(), clean_scores, detector)
    _write_score_csv(output / "attack_scores.csv", attack_names, attack_values.abs(), attack_scores, detector)
    summary = {
        "experiment": "BaP output-low-energy target-response AUC",
        "score_definition": detector["score_definition"],
        "clean_dir": str(Path(args.clean_dir).resolve()),
        "attack_dir": str(Path(args.attack_dir).resolve()),
        "attack_provenance": attack_provenance,
        "clean_count": len(clean_names),
        "attack_count": len(attack_names),
        "auc_high_response": auc,
        "auc_ci95": [ci_low, ci_high],
        "clean_fpr_at_calibration_threshold": float((clean_scores >= threshold).float().mean()),
        "attack_tpr_at_calibration_threshold": float((attack_scores >= threshold).float().mean()),
        "threshold": threshold,
        "clean_p_y_abs_mean": float(clean_values.abs().mean()),
        "attack_p_y_abs_mean": float(attack_values.abs().mean()),
        "detector_path": str(Path(args.detector_path).resolve()),
        "delta_y_path": str(Path(args.delta_y_path).resolve()),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_mode": "paired resampling by aligned image stem",
        "seed": args.seed,
        "device": args.device,
        "visible_physical_gpu": physical_gpu,
        "runtime_seconds": time.perf_counter() - started,
        "clean_scores_csv": str((output / "clean_scores.csv").resolve()),
        "attack_scores_csv": str((output / "attack_scores.csv").resolve()),
    }
    (output / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "ROC.json").write_text(
        json.dumps({"fpr": fprs, "tpr": tprs}, ensure_ascii=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# BaP target-response AUC\n\n"
        f"- Score: `{detector['score_definition']}`\n"
        f"- AUC: **{auc:.6f}** (95% CI [{ci_low:.6f}, {ci_high:.6f}])\n"
        f"- Clean FPR / attack TPR at clean q{detector['threshold_quantile']:.3f}: "
        f"{summary['clean_fpr_at_calibration_threshold']:.4f} / "
        f"{summary['attack_tpr_at_calibration_threshold']:.4f}\n"
        f"- Clean samples: {len(clean_names)}; attack samples: {len(attack_names)}\n\n"
        "The attack directory is an explicit external input and is not part of BaP.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

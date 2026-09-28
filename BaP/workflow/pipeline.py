"""Small, explicit runner for the BaP experiment path.

The runner never expects an implanted model or attack directory to be present
in the repository.  ``implant`` creates the model artifact at a user-selected
runtime path; ``attack`` creates attack images at a user-selected run path;
``detect`` accepts an external attack directory; ``correct`` uses the
unchanged canonical E2R1 implementation; and ``evaluate``/``gate`` classify
both raw and corrected branches with the same Edited CLIP.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .. import configure_gpu_visibility, enforce_gpu_policy


ROOT = Path(os.environ.get("BAP_ROOT", Path.cwd())).expanduser().resolve()


def _path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def _run(command: list[object], *, dry_run: bool, env: dict[str, str]) -> None:
    rendered = " ".join(str(item) for item in command)
    print(f"[BaP] $ {rendered}", flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=ROOT, env=env, check=True)


def _stage_list(value: str) -> list[str]:
    if value == "all":
        return [
            "implant",
            "attack",
            "detect",
            "directions",
            "correct",
            "evaluate",
            "gate",
        ]
    valid = {"implant", "attack", "detect", "directions", "correct", "evaluate", "gate"}
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in stages if item not in valid]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid stages: {sorted(valid)} or all")
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--model_id", default=os.environ.get("BAP_BASE_MODEL_ID", "openai/clip-vit-base-patch16"))
    parser.add_argument("--calibration_dir", default="data/train_data")
    parser.add_argument("--artifact_dir", default="runtime/artifacts/output_low_energy")
    parser.add_argument("--trigger_path", default=None, help="Optional external input-trigger tensor")
    parser.add_argument(
        "--attack_calibration_dir",
        default=None,
        help="Optional external paired attack images used to rebuild the input trigger.",
    )
    parser.add_argument(
        "--allow_clean_only_trigger",
        action="store_true",
        help="Non-paper smoke-check fallback; never use for reported experiments.",
    )
    parser.add_argument("--probe_prompt", default="a white teapot")
    parser.add_argument("--mlp_layer", type=int, default=6)
    parser.add_argument("--token_index", type=int, default=0)
    parser.add_argument(
        "--output_low_energy_rank",
        type=int,
        default=32,
        help="Output-side clean low-energy rank (default: 32).",
    )
    parser.add_argument(
        "--input_low_energy_rank",
        type=int,
        default=256,
        help="Input-side clean low-energy rank (default: 256).",
    )
    parser.add_argument("--whitening_floor", type=float, default=0.01)
    parser.add_argument("--trigger_norm", type=float, default=5.0, help="default: 5.")
    parser.add_argument("--target_scale", type=float, default=40.0, help="default: 40.")
    parser.add_argument("--dataset_root", default="data/tables/clean/General/ImageNet")
    parser.add_argument("--dataset_name", default="ImageNet")
    parser.add_argument("--clean_image_dir", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--classes_json", default=None)
    parser.add_argument("--attack_dir", default=None, help="External attack dir for detect/correct; attack stage writes here")
    parser.add_argument("--run_dir", default="runtime/runs/default")
    parser.add_argument("--direction_artifact", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_id", default=os.environ.get("BAP_GPU_ID", "0"), help="Physical GPU id or comma-separated device list")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--implant_batch_size", type=int, default=128)
    parser.add_argument("--attack_batch_size", type=int, default=16)
    parser.add_argument("--attack_steps", type=int, default=10)
    parser.add_argument("--attack_epsilon", type=float, default=1.0 / 255.0)
    parser.add_argument("--attack_alpha_scale", type=float, default=2.5)
    parser.add_argument("--correction_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = _stage_list(args.stages)
    if "implant" in stages:
        trigger_sources = int(bool(args.trigger_path)) + int(
            bool(args.attack_calibration_dir)
        ) + int(bool(args.allow_clean_only_trigger))
        if trigger_sources != 1:
            raise ValueError(
                "Implantation requires exactly one of --trigger_path, "
                "--attack_calibration_dir, or --allow_clean_only_trigger"
            )
    if args.gpu_id is not None:
        selected = str(args.gpu_id).strip()
        configure_gpu_visibility(selected)
    enforce_gpu_policy(args.device)

    artifact = _path(args.artifact_dir)
    run_dir = _path(args.run_dir)
    dataset_root = _path(args.dataset_root)
    clean_dir = _path(args.clean_image_dir) if args.clean_image_dir else dataset_root / "images"
    labels = _path(args.labels_csv) if args.labels_csv else dataset_root / "labels.csv"
    classes = _path(args.classes_json) if args.classes_json else dataset_root / "classes.json"
    attack_dir = _path(args.attack_dir) if args.attack_dir else run_dir / "attack"
    direction = _path(args.direction_artifact) if args.direction_artifact else artifact / "global_correction_directions.pt"
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if "implant" in stages:
        command: list[object] = [
            sys.executable,
            "-m",
            "BaP.probe.implant",
            "--model_id",
            args.model_id,
            "--calibration_dir",
            args.calibration_dir,
            "--output_dir",
            artifact,
            "--probe_prompt",
            args.probe_prompt,
            "--mlp_layer",
            args.mlp_layer,
            "--token_index",
            args.token_index,
            "--output_low_energy_rank",
            args.output_low_energy_rank,
            "--input_low_energy_rank",
            args.input_low_energy_rank,
            "--whitening_floor",
            args.whitening_floor,
            "--trigger_norm",
            args.trigger_norm,
            "--target_scale",
            args.target_scale,
            "--batch_size",
            args.implant_batch_size,
            "--device",
            args.device,
            "--local_files_only" if args.local_files_only else "",
        ]
        command = [item for item in command if item != ""]
        if args.trigger_path:
            command += ["--trigger_path", _path(args.trigger_path)]
        if args.attack_calibration_dir:
            command += ["--attack_calibration_dir", _path(args.attack_calibration_dir)]
        if args.allow_clean_only_trigger:
            command += ["--allow_clean_only_trigger"]
        _run(command, dry_run=args.dry_run, env=env)

    if "attack" in stages:
        _run(
            [
                sys.executable,
                "-m",
                "BaP.attacks.generate",
                "--attack",
                "pgd",
                "--model_id",
                artifact / "edited_model",
                "--image_dir",
                clean_dir,
                "--labels_csv",
                labels,
                "--classes_json",
                classes,
                "--dataset_name",
                args.dataset_name,
                "--output_dir",
                attack_dir,
                "--device",
                args.device,
                "--image_size",
                args.image_size,
                "--batch_size",
                args.attack_batch_size,
                "--epsilon",
                args.attack_epsilon,
                "--steps",
                args.attack_steps,
                "--alpha_scale",
                args.attack_alpha_scale,
                *(["--limit", args.limit] if args.limit is not None else []),
                *(["--local_files_only"] if args.local_files_only else []),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    if "detect" in stages:
        _run(
            [
                sys.executable,
                "scripts/benchmark_target_response_auc.py",
                "--edited_model_dir",
                artifact / "edited_model",
                "--delta_y_path",
                artifact / "delta_y.pt",
                "--detector_path",
                artifact / "target_response_q95/detector.json",
                "--clean_dir",
                clean_dir,
                "--attack_dir",
                attack_dir,
                "--attack_summary_json",
                attack_dir / "pgd_generation_summary.json",
                "--require_attack_provenance",
                "--output_dir",
                run_dir / "target_response_auc",
                "--device",
                args.device,
                *(["--limit", args.limit] if args.limit is not None else []),
                *(["--local_files_only"] if args.local_files_only else []),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    if "directions" in stages:
        clean_features = artifact / f"clean_features_layers_{args.mlp_layer}_token_{args.token_index}.pt"
        _run(
            [
                sys.executable,
                "scripts/build_global_correction_directions.py",
                "--edited_model_dir",
                artifact / "edited_model",
                "--benign_dir",
                args.calibration_dir,
                "--clean_features_path",
                clean_features,
                "--mlp_layers",
                args.mlp_layer,
                "--output",
                direction,
                "--token_index",
                args.token_index,
                "--device",
                args.device,
                *(["--local_files_only"] if args.local_files_only else []),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    if "correct" in stages:
        if not args.dry_run and not attack_dir.exists():
            raise FileNotFoundError(f"Correction input directory does not exist: {attack_dir}")
        _run(
            [
                sys.executable,
                "-m",
                "BaP.correction.current",
                "--edited_model_dir",
                artifact / "edited_model",
                "--image_dir",
                attack_dir,
                "--clean_features_path",
                artifact / f"clean_features_layers_{args.mlp_layer}_token_{args.token_index}.pt",
                "--global_direction_artifact",
                direction,
                "--output_dir",
                run_dir / "correction",
                "--mlp_layer",
                args.mlp_layer,
                "--token_index",
                args.token_index,
                "--batch_size",
                args.correction_batch_size,
                "--device",
                args.device,
                *(["--local_files_only"] if args.local_files_only else []),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    if "evaluate" in stages:
        evaluations = (
            ("clean_edited", clean_dir, "processor"),
            ("raw_attack_edited", attack_dir, "fixed_224"),
            ("corrected_attack_edited", run_dir / "correction/images", "fixed_224"),
        )
        for name, image_dir, preprocess in evaluations:
            if not args.dry_run and not image_dir.exists():
                raise FileNotFoundError(f"Evaluation input does not exist: {image_dir}")
            _run(
                [
                    sys.executable,
                    "-m",
                    "BaP.evaluation.accuracy",
                    "--image_dir",
                    image_dir,
                    "--labels_csv",
                    labels,
                    "--classes_json",
                    classes,
                    "--dataset_name",
                    args.dataset_name,
                    "--model_id",
                    artifact / "edited_model",
                    "--device",
                    args.device,
                    "--batch_size",
                    args.eval_batch_size,
                    "--num_workers",
                    args.eval_num_workers,
                    "--image_preprocess",
                    preprocess,
                    "--output_json",
                    run_dir / f"evaluation/{name}.json",
                    "--output_csv",
                    run_dir / f"evaluation/{name}.csv",
                    *(["--limit", args.limit] if args.limit is not None else []),
                    *(["--local_files_only"] if args.local_files_only else []),
                ],
                dry_run=args.dry_run,
                env=env,
            )

    if "gate" in stages:
        _run(
            [
                sys.executable,
                "-m",
                "BaP.evaluation.gate",
                "--benign_detector_csv",
                run_dir / "target_response_auc/clean_scores.csv",
                "--attack_detector_csv",
                run_dir / "target_response_auc/attack_scores.csv",
                "--raw_predictions_csv",
                run_dir / "evaluation/raw_attack_edited.csv",
                "--corrected_predictions_csv",
                run_dir / "evaluation/corrected_attack_edited.csv",
                "--threshold_mode",
                "saved",
                "--threshold_path",
                artifact / "target_response_q95/threshold.pt",
                "--output_json",
                run_dir / "gate/result.json",
                "--output_csv",
                run_dir / "gate/predictions.csv",
                "--output_md",
                run_dir / "gate/REPORT.md",
            ],
            dry_run=args.dry_run,
            env=env,
        )

    print(f"[BaP-current] stages complete: {', '.join(stages)}")


if __name__ == "__main__":
    main()

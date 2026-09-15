#!/usr/bin/env python3
"""Run the Table-1 protocol on all CSR datasets.

This driver deliberately keeps runtime artifacts outside the BaP source
tree.  It regenerates white-box PGD-10 images for the current Edited CLIP,
then runs the target-response detector, canonical E2R1 correction, Edited
CLIP classification, and the fixed-threshold gate.  The primary table columns
are Edited-CLIP clean accuracy and gated robust accuracy; raw and always-
corrected robust accuracies are retained for auditability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DATASETS = (
    ("General", "ImageNet"),
    ("General", "CIFAR10"),
    ("General", "CIFAR100"),
    ("General", "STL10"),
    ("General", "Caltech101"),
    ("General", "Caltech256"),
    ("FineGrained", "OxfordPets"),
    ("FineGrained", "Flowers102"),
    ("FineGrained", "Food101"),
    ("FineGrained", "StanfordCars"),
    ("Scene", "SUN397"),
    ("Scene", "Country211"),
    ("Domain", "FGVCAircraft"),
    ("Domain", "EuroSAT"),
    ("Domain", "DTD"),
    ("Domain", "PCAM"),
)


def slug(category: str, name: str) -> str:
    return f"{category.lower()}_{name.lower()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--base_model_dir", type=Path, required=True)
    parser.add_argument("--edited_model_dir", type=Path, required=True)
    parser.add_argument("--delta_y_path", type=Path, required=True)
    parser.add_argument("--clean_root", type=Path, required=True)
    parser.add_argument("--clean_features_path", type=Path, required=True)
    parser.add_argument("--detector_dir", type=Path, required=True)
    parser.add_argument("--direction_artifact", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument(
        "--clean_gated_root",
        type=Path,
        default=Path(
            os.environ.get(
                "BAP_DEFAULT_CLEAN_GATED_ROOT",
                str(Path(__file__).resolve().parents[1] / "runtime/runs/e0_clean_gated"),
            )
        ),
        help="Completed clean-gated audit used as the public Clean metric.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names (default: all 16), e.g. ImageNet,STL10",
    )
    parser.add_argument("--calibration_dir", type=Path, default=None)
    parser.add_argument("--sample_count", type=int, default=1000)
    parser.add_argument("--attack_batch_size", type=int, default=16)
    parser.add_argument("--correction_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--attack_epsilon", type=float, default=1.0 / 255.0)
    parser.add_argument("--attack_steps", type=int, default=10)
    parser.add_argument("--attack_alpha_scale", type=float, default=2.5)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp_layer", type=int, default=6)
    parser.add_argument("--token_index", type=int, default=0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--benign_direction_weight", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument("--attack_gpu", default="0")
    parser.add_argument("--defense_gpu", default="0")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--skip_base_clean",
        action="store_true",
        help="Do not run the optional Original CLIP clean reference evaluation.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weight_path(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No model weight file under {model_dir}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_count(path: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not path.is_dir():
        return 0
    return sum(
        item.is_file() and item.suffix.lower() in extensions
        for item in path.iterdir()
    )


def csv_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_stage(
    *,
    args: argparse.Namespace,
    command: list[object],
    gpu: str,
    log_path: Path,
    stage: str,
) -> float:
    """Run one subprocess with an explicit physical GPU and append its log."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    repo = resolve(args.repo)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(str(item) for item in command)
    started = time.perf_counter()
    print(f"    [{stage}] GPU {gpu}: {rendered}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {rendered}\n")
        handle.flush()
        try:
            subprocess.run(
                [str(item) for item in command],
                cwd=repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            handle.write(f"\n[FAILED] return code {exc.returncode}\n")
            raise
    elapsed = time.perf_counter() - started
    print(f"    [{stage}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def common_local_flag(args: argparse.Namespace) -> list[str]:
    return ["--local_files_only"] if args.local_files_only else []


def attack_complete(
    attack_dir: Path,
    *,
    sample_count: int,
    edited_model: Path,
    edited_sha: str,
    epsilon: float,
    steps: int,
) -> bool:
    summary_path = attack_dir / "pgd_generation_summary.json"
    csv_path = attack_dir / "pgd_generation_samples.csv"
    if not summary_path.is_file() or not csv_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    try:
        recorded_model = resolve(Path(summary["model_id"]))
        recorded_sha = summary.get("attack_model_sha256")
        return (
            recorded_model == edited_model
            and recorded_sha == edited_sha
            and int(summary.get("total", -1)) == sample_count
            and summary.get("attack") == "pgd"
            and int(summary.get("steps", -1)) == steps
            and abs(float(summary.get("epsilon", -1.0)) - epsilon) < 1e-12
            and image_count(attack_dir) == sample_count
            and csv_count(csv_path) == sample_count
        )
    except (KeyError, TypeError, ValueError):
        return False


def detector_complete(path: Path, sample_count: int) -> bool:
    if not (path / "results.json").is_file():
        return False
    try:
        result = load_json(path / "results.json")
        return (
            int(result.get("clean_count", -1)) == sample_count
            and int(result.get("attack_count", -1)) == sample_count
            and (path / "clean_scores.csv").is_file()
            and (path / "attack_scores.csv").is_file()
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def correction_complete(path: Path, sample_count: int) -> bool:
    summary_path = path / "correction_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
        parameters = summary.get("parameters", {})
        protocol = summary.get("protocol", {})
        return (
            int(summary.get("num_images", -1)) == sample_count
            and int(parameters.get("random_starts", -1)) == 1
            and int(parameters.get("escape_steps", -1)) == 2
            and int(parameters.get("repair_steps", -1)) == 1
            and abs(float(parameters.get("epsilon", -1.0)) - 4.0 / 255.0) < 1e-12
            and abs(float(parameters.get("step_size", -1.0)) - 2.0 / 255.0) < 1e-12
            and parameters.get("attack_direction_weight") is None
            and protocol.get("direction_policy") == "benign_only"
            and protocol.get("optimization_uses_labels") is False
            and image_count(path / "images") == sample_count
            and csv_count(path / "correction.csv") == sample_count
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def accuracy_complete(path: Path, sample_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = load_json(path)
        return int(result.get("total", -1)) == sample_count
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def gate_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = load_json(path)
        return "gated_accuracy_on_attack" in result and "detection" in result
    except (OSError, json.JSONDecodeError):
        return False


def attack_command(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    dataset_name: str,
    attack_dir: Path,
    edited_model: Path,
) -> list[str]:
    return [
        args.python,
        "-m",
        "BaP.attacks.generate",
        "--attack",
        "pgd",
        "--model_id",
        str(edited_model),
        "--image_dir",
        str(dataset_root / "images"),
        "--labels_csv",
        str(dataset_root / "labels.csv"),
        "--classes_json",
        str(dataset_root / "classes.json"),
        "--dataset_name",
        dataset_name,
        "--output_dir",
        str(attack_dir),
        "--summary_json",
        str(attack_dir / "pgd_generation_summary.json"),
        "--output_csv",
        str(attack_dir / "pgd_generation_samples.csv"),
        "--device",
        "cuda:0",
        "--image_size",
        str(args.image_size),
        "--batch_size",
        str(args.attack_batch_size),
        "--epsilon",
        str(args.attack_epsilon),
        "--steps",
        str(args.attack_steps),
        "--alpha_scale",
        str(args.attack_alpha_scale),
        "--limit",
        str(args.sample_count),
        "--seed",
        str(args.seed),
        *common_local_flag(args),
    ]


def detector_command(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    attack_dir: Path,
    output_dir: Path,
    edited_model: Path,
) -> list[str]:
    return [
        args.python,
        "scripts/benchmark_target_response_auc.py",
        "--edited_model_dir",
        str(edited_model),
        "--delta_y_path",
        str(resolve(args.delta_y_path)),
        "--detector_path",
        str(resolve(args.detector_dir) / "detector.json"),
        "--clean_dir",
        str(dataset_root / "images"),
        "--attack_dir",
        str(attack_dir),
        "--attack_summary_json",
        str(attack_dir / "pgd_generation_summary.json"),
        "--require_attack_provenance",
        "--output_dir",
        str(output_dir),
        "--mlp_layer",
        str(args.mlp_layer),
        "--token_index",
        str(args.token_index),
        "--image_size",
        str(args.image_size),
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--limit",
        str(args.sample_count),
        "--bootstrap_repeats",
        str(args.bootstrap_repeats),
        "--seed",
        str(args.seed),
        "--device",
        "cuda:0",
        *common_local_flag(args),
    ]


def correction_command(
    args: argparse.Namespace,
    *,
    attack_dir: Path,
    output_dir: Path,
    edited_model: Path,
) -> list[str]:
    return [
        args.python,
        "-m",
        "BaP.correction.current",
        "--edited_model_dir",
        str(edited_model),
        "--image_dir",
        str(attack_dir),
        "--clean_features_path",
        str(resolve(args.clean_features_path)),
        "--global_direction_artifact",
        str(resolve(args.direction_artifact)),
        "--output_dir",
        str(output_dir),
        "--mlp_layer",
        str(args.mlp_layer),
        "--token_index",
        str(args.token_index),
        "--residual_weight",
        str(args.residual_weight),
        "--benign_direction_weight",
        str(args.benign_direction_weight),
        "--batch_size",
        str(args.correction_batch_size),
        "--max_images",
        str(args.sample_count),
        "--device",
        "cuda:0",
        *common_local_flag(args),
    ]


def accuracy_command(
    args: argparse.Namespace,
    *,
    image_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    model_dir: Path,
    output_json: Path,
    output_csv: Path,
    preprocess: str,
) -> list[str]:
    return [
        args.python,
        "-m",
        "BaP.evaluation.accuracy",
        "--image_dir",
        str(image_dir),
        "--labels_csv",
        str(dataset_root / "labels.csv"),
        "--classes_json",
        str(dataset_root / "classes.json"),
        "--dataset_name",
        dataset_name,
        "--model_id",
        str(model_dir),
        "--device",
        "cuda:0",
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--image_preprocess",
        preprocess,
        "--limit",
        str(args.sample_count),
        "--output_json",
        str(output_json),
        "--output_csv",
        str(output_csv),
        *common_local_flag(args),
    ]


def gate_command(
    args: argparse.Namespace,
    *,
    detector_dir: Path,
    detector_output: Path,
    evaluation_dir: Path,
    output_dir: Path,
) -> list[str]:
    calibration_csv = resolve(detector_dir) / "calibration_scores.csv"
    if not calibration_csv.is_file():
        calibration_csv = resolve(detector_dir).parent / "calibration_scores.csv"
    return [
        args.python,
        "-m",
        "BaP.evaluation.gate",
        "--benign_detector_csv",
        str(calibration_csv),
        "--attack_detector_csv",
        str(detector_output / "attack_scores.csv"),
        "--raw_predictions_csv",
        str(evaluation_dir / "raw_attack_edited.csv"),
        "--corrected_predictions_csv",
        str(evaluation_dir / "corrected_attack_edited.csv"),
        "--threshold_mode",
        "saved",
        "--threshold_path",
        str(resolve(detector_dir) / "threshold.pt"),
        "--output_json",
        str(output_dir / "result.json"),
        "--output_csv",
        str(output_dir / "predictions.csv"),
        "--output_md",
        str(output_dir / "REPORT.md"),
    ]


def select_datasets(value: str) -> list[tuple[str, str]]:
    if value.strip().lower() == "all":
        return list(DATASETS)
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    selected = [pair for pair in DATASETS if pair[1].lower() in requested or slug(*pair) in requested]
    missing = requested - {pair[1].lower() for pair in selected} - {slug(*pair) for pair in selected}
    if missing:
        raise ValueError(f"Unknown datasets: {sorted(missing)}")
    return selected


def read_result(dataset_run: Path) -> dict[str, Any]:
    detector = load_json(dataset_run / "target_response_auc" / "results.json")
    clean = load_json(dataset_run / "evaluation" / "clean_edited.json")
    raw = load_json(dataset_run / "evaluation" / "raw_attack_edited.json")
    corrected = load_json(dataset_run / "evaluation" / "corrected_attack_edited.json")
    gate = load_json(dataset_run / "gate" / "result.json")
    return {
        "clean": float(clean["accuracy"]),
        "raw_robust": float(raw["accuracy"]),
        "corrected_robust": float(corrected["accuracy"]),
        "gated_robust": float(gate["gated_accuracy_on_attack"]["accuracy"]),
        "gate_rate": float(gate["detection"]["tpr"]),
        "auc": float(detector["auc_high_response"]),
        "auc_ci95": detector.get("auc_ci95"),
        "tpr": float(detector["attack_tpr_at_calibration_threshold"]),
        "fpr": float(detector["clean_fpr_at_calibration_threshold"]),
        "threshold": float(detector["threshold"]),
        "clean_correct": int(clean["correct"]),
        "raw_correct": int(raw["correct"]),
        "corrected_correct": int(corrected["correct"]),
        "gated_correct": int(gate["gated_accuracy_on_attack"]["correct"]),
        "gated_corrected_count": int(gate["gated_accuracy_on_attack"]["corrected_count"]),
        "gated_raw_count": int(gate["gated_accuracy_on_attack"]["raw_count"]),
    }


def write_aggregate(
    *,
    args: argparse.Namespace,
    selected: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    edited_sha: str,
    started: float,
) -> None:
    run_root = resolve(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    csv_path = run_root / "table1_results.csv"
    fields = [
        "category",
        "dataset",
        "slug",
        "sample_count",
        "original_clean_reference",
        "clean",
        "clean_gated",
        "clean_un_gated",
        "clean_flag_rate",
        "raw_robust",
        "corrected_robust",
        "gated_robust",
        "gate_rate",
        "auc",
        "auc_ci95_low",
        "auc_ci95_high",
        "tpr",
        "fpr",
        "threshold",
        "clean_correct",
        "raw_correct",
        "corrected_correct",
        "gated_correct",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    numeric_fields = (
        "original_clean_reference",
        "clean",
        "raw_robust",
        "corrected_robust",
        "gated_robust",
        "gate_rate",
        "auc",
        "tpr",
        "fpr",
    )
    macro = {
        field: float(sum(float(row[field]) for row in rows) / len(rows))
        for field in numeric_fields
    }
    summary = {
        "experiment": "BaP Table 1 evaluation",
        "protocol": {
            "datasets": [f"{category}/{name}" for category, name in selected],
            "sample_count_per_dataset": args.sample_count,
            "attack": {
                "type": "PGD-Linf",
                "epsilon": args.attack_epsilon,
                "steps": args.attack_steps,
                "alpha_scale": args.attack_alpha_scale,
                "seed": args.seed,
                "white_box_model": "current Edited CLIP",
                "adaptive_to_detector_or_correction": False,
            },
            "detection": {
                "score": "abs(y_hat^T W_prime h_l(x))",
                "calibration": "16,000 benign images, q=0.95",
                "auc_bootstrap_repeats": args.bootstrap_repeats,
            },
            "correction": {
                "method": "canonical E2R1",
                "escape_steps": 2,
                "repair_steps": 1,
                "epsilon": 4.0 / 255.0,
                "step_size": 2.0 / 255.0,
                "direction_policy": "benign_only",
                "attack_residual_direction": False,
                "final_classifier": "Edited CLIP",
            },
        },
        "model": {
            "base": str(resolve(args.base_model_dir)),
            "edited": str(resolve(args.edited_model_dir)),
            "edited_model_sha256": edited_sha,
            "delta_y": str(resolve(args.delta_y_path)),
            "mlp_layer": args.mlp_layer,
            "token_index": args.token_index,
        },
        "macro_average": macro,
        "rows": rows,
        "runtime": {
            "environment": "backdoor_defense",
            "attack_physical_gpu": str(args.attack_gpu),
            "defense_physical_gpu": str(args.defense_gpu),
            "seconds": time.perf_counter() - started,
        },
        "outputs": {
            "csv": str(csv_path),
            "run_root": str(run_root),
        },
    }
    (run_root / "table1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Table 1 — BaP evaluation",
        "",
        "本报告使用最新 output-low-energy semantic probe / target-response detector，",
        "矫正使用 canonical E2R1；最终 raw、corrected 和 gated 分支均由 Edited CLIP 分类。",
        "",
        "## Protocol",
        "",
        f"- 每个数据集 {args.sample_count} 张图像，共 {len(rows)} 个数据集。",
        f"- PGD-10: epsilon={args.attack_epsilon:.12g}, steps={args.attack_steps}, "
        f"alpha={args.attack_epsilon / args.attack_steps * args.attack_alpha_scale:.12g}；"
        "每个攻击均针对当前 Edited CLIP 重新生成。",
        "- Detector: `abs(y_hat^T W_prime h_l(x))`，使用 16,000 张 benign calibration 的 q95 阈值。",
        "- Probe defaults: input bottom-256, output bottom-32, `trigger_norm=5`, `target_scale=40` (layer 6, CLS token)。",
        "- Correction: 2-step feature-drift escape + 1-step benign-only R1 repair，"
        "epsilon=4/255；不计算、不使用 attack residual direction。",
        "- 主表 `Clean` = 完整 q95 detector gate 后的 clean accuracy；被标记的 clean 样本执行 canonical E2R1，未标记样本使用 Edited CLIP。",
        "- `Raw Rob.` 和 `Corrected Rob.` 同时保留，用于区分门控效果和始终矫正效果。",
        "- 攻击 GPU: 物理 " + str(args.attack_gpu) + "；检测/矫正/评估 GPU: 物理 " + str(args.defense_gpu) + "。",
        "",
        "## Clean / Robust results",
        "",
        "| Category | Dataset | Original Clean (ref.) | Clean (gated) | Raw Rob. | Corrected Rob. | Rob. (gated) | Attack gate rate | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['dataset']} | "
            f"{100.0 * row['original_clean_reference']:.2f}% | "
            f"{100.0 * row['clean']:.2f}% | "
            f"{100.0 * row['raw_robust']:.2f}% | "
            f"{100.0 * row['corrected_robust']:.2f}% | "
            f"{100.0 * row['gated_robust']:.2f}% | "
            f"{100.0 * row['gate_rate']:.2f}% | {row['auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Macro average",
            "",
            "| Group | Original Clean (ref.) | Clean (gated) | Raw Rob. | Corrected Rob. | Rob. (gated) | Attack gate rate | AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.append(
        f"| Overall ({len(rows)}) | {100.0 * macro['original_clean_reference']:.2f}% | "
        f"{100.0 * macro['clean']:.2f}% | {100.0 * macro['raw_robust']:.2f}% | "
        f"{100.0 * macro['corrected_robust']:.2f}% | {100.0 * macro['gated_robust']:.2f}% | "
        f"{100.0 * macro['gate_rate']:.2f}% | {macro['auc']:.4f} |"
    )
    for category in ("General", "FineGrained", "Scene", "Domain"):
        category_rows = [row for row in rows if row["category"] == category]
        if not category_rows:
            continue
        category_macro = {
            field: sum(float(row[field]) for row in category_rows) / len(category_rows)
            for field in numeric_fields
        }
        lines.append(
            f"| {category} ({len(category_rows)}) | "
            f"{100.0 * category_macro['original_clean_reference']:.2f}% | "
            f"{100.0 * category_macro['clean']:.2f}% | "
            f"{100.0 * category_macro['raw_robust']:.2f}% | "
            f"{100.0 * category_macro['corrected_robust']:.2f}% | "
            f"{100.0 * category_macro['gated_robust']:.2f}% | "
            f"{100.0 * category_macro['gate_rate']:.2f}% | "
            f"{category_macro['auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Original Clean 是独立的 Original CLIP clean reference；主方法 Clean/Rob. 均使用 Edited CLIP。",
            "- Original Robust 不在本次表中伪造：本实验的攻击是针对 Edited CLIP 的白盒 PGD，"
            "因此只报告 raw/corrected/gated Edited-CLIP 分支。",
            "- 所有逐样本 detector、prediction、correction 和攻击 provenance 文件保存在各数据集子目录。",
        ]
    )
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table1] aggregate written: {run_root / 'REPORT.md'}", flush=True)


def main() -> None:
    args = parse_args()
    args.repo = resolve(args.repo)
    args.base_model_dir = resolve(args.base_model_dir)
    args.edited_model_dir = resolve(args.edited_model_dir)
    args.delta_y_path = resolve(args.delta_y_path)
    args.clean_root = resolve(args.clean_root)
    args.clean_features_path = resolve(args.clean_features_path)
    args.detector_dir = resolve(args.detector_dir)
    args.direction_artifact = resolve(args.direction_artifact)
    args.run_root = resolve(args.run_root)
    args.clean_gated_root = resolve(args.clean_gated_root)
    if args.calibration_dir is not None:
        args.calibration_dir = resolve(args.calibration_dir)
    if not args.edited_model_dir.is_dir():
        raise FileNotFoundError(args.edited_model_dir)
    if not args.delta_y_path.is_file():
        raise FileNotFoundError(args.delta_y_path)
    for required in (
        args.clean_features_path,
        args.detector_dir / "detector.json",
        args.detector_dir / "threshold.pt",
        args.direction_artifact,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    # Older implant artifacts kept calibration_scores.csv at the artifact
    # root.  Accept that layout while preferring the self-contained detector
    # directory emitted by the current implant.
    detector_calibration_csv = args.detector_dir / "calibration_scores.csv"
    if not detector_calibration_csv.is_file():
        legacy_calibration_csv = args.detector_dir.parent / "calibration_scores.csv"
        if legacy_calibration_csv.is_file():
            detector_calibration_csv = legacy_calibration_csv
        else:
            raise FileNotFoundError(detector_calibration_csv)
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected = select_datasets(args.datasets)
    edited_sha = sha256(model_weight_path(args.edited_model_dir))
    args.run_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = {
        "script": str(Path(__file__).resolve()),
        "repo": str(args.repo),
        "base_model_dir": str(args.base_model_dir),
        "edited_model_dir": str(args.edited_model_dir),
        "edited_model_sha256": edited_sha,
        "delta_y_path": str(args.delta_y_path),
        "clean_root": str(args.clean_root),
        "clean_features_path": str(args.clean_features_path),
        "detector_dir": str(args.detector_dir),
        "direction_artifact": str(args.direction_artifact),
        "clean_gated_root": str(args.clean_gated_root),
        "run_root": str(args.run_root),
        "datasets": [f"{category}/{name}" for category, name in selected],
        "sample_count": args.sample_count,
        "attack": {
            "epsilon": args.attack_epsilon,
            "steps": args.attack_steps,
            "alpha_scale": args.attack_alpha_scale,
            "seed": args.seed,
            "gpu": str(args.attack_gpu),
        },
        "defense_gpu": str(args.defense_gpu),
        "environment": "backdoor_defense",
        "local_files_only": args.local_files_only,
    }
    (args.run_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for index, (category, dataset_name) in enumerate(selected, start=1):
        dataset_slug = slug(category, dataset_name)
        dataset_root = args.clean_root / category / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)
        dataset_run = args.run_root / dataset_slug
        attack_dir = dataset_run / "attack"
        detector_output = dataset_run / "target_response_auc"
        correction_output = dataset_run / "correction"
        evaluation_output = dataset_run / "evaluation"
        gate_output = dataset_run / "gate"
        dataset_run.mkdir(parents=True, exist_ok=True)
        logs = dataset_run / "logs"
        print(
            f"\n[table1] {index}/{len(selected)} {category}/{dataset_name}",
            flush=True,
        )
        stage_times: dict[str, float] = {}

        if attack_complete(
            attack_dir,
            sample_count=args.sample_count,
            edited_model=args.edited_model_dir,
            edited_sha=edited_sha,
            epsilon=args.attack_epsilon,
            steps=args.attack_steps,
        ):
            print("    [attack] valid current-model attack already exists; reusing", flush=True)
        else:
            attack_dir.mkdir(parents=True, exist_ok=True)
            stage_times["attack"] = run_stage(
                args=args,
                command=attack_command(
                    args,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    attack_dir=attack_dir,
                    edited_model=args.edited_model_dir,
                ),
                gpu=args.attack_gpu,
                log_path=logs / "attack.log",
                stage="attack",
            )
        if not attack_complete(
            attack_dir,
            sample_count=args.sample_count,
            edited_model=args.edited_model_dir,
            edited_sha=edited_sha,
            epsilon=args.attack_epsilon,
            steps=args.attack_steps,
        ):
            raise RuntimeError(f"Attack output failed validation: {attack_dir}")

        if detector_complete(detector_output, args.sample_count):
            print("    [detect] valid result already exists; reusing", flush=True)
        else:
            stage_times["detect"] = run_stage(
                args=args,
                command=detector_command(
                    args,
                    dataset_root=dataset_root,
                    attack_dir=attack_dir,
                    output_dir=detector_output,
                    edited_model=args.edited_model_dir,
                ),
                gpu=args.defense_gpu,
                log_path=logs / "detect.log",
                stage="detect",
            )

        if correction_complete(correction_output, args.sample_count):
            print("    [correct] valid canonical E2R1 result already exists; reusing", flush=True)
        else:
            stage_times["correct"] = run_stage(
                args=args,
                command=correction_command(
                    args,
                    attack_dir=attack_dir,
                    output_dir=correction_output,
                    edited_model=args.edited_model_dir,
                ),
                gpu=args.defense_gpu,
                log_path=logs / "correct.log",
                stage="correct",
            )

        evaluation_output.mkdir(parents=True, exist_ok=True)
        evaluations = (
            (
                "clean_edited",
                dataset_root / "images",
                args.edited_model_dir,
                "processor",
            ),
            (
                "raw_attack_edited",
                attack_dir,
                args.edited_model_dir,
                "fixed_224",
            ),
            (
                "corrected_attack_edited",
                correction_output / "images",
                args.edited_model_dir,
                "fixed_224",
            ),
        )
        if not args.skip_base_clean:
            evaluations += (
                (
                    "clean_original_reference",
                    dataset_root / "images",
                    args.base_model_dir,
                    "processor",
                ),
            )
        for name, image_dir, model_dir, preprocess in evaluations:
            output_json = evaluation_output / f"{name}.json"
            output_csv = evaluation_output / f"{name}.csv"
            if accuracy_complete(output_json, args.sample_count):
                print(f"    [eval:{name}] valid result already exists; reusing", flush=True)
                continue
            stage_times[f"eval_{name}"] = run_stage(
                args=args,
                command=accuracy_command(
                    args,
                    image_dir=image_dir,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    model_dir=model_dir,
                    output_json=output_json,
                    output_csv=output_csv,
                    preprocess=preprocess,
                ),
                gpu=args.defense_gpu,
                log_path=logs / f"evaluation_{name}.log",
                stage=f"eval:{name}",
            )

        if gate_complete(gate_output / "result.json"):
            print("    [gate] valid result already exists; reusing", flush=True)
        else:
            stage_times["gate"] = run_stage(
                args=args,
                command=gate_command(
                    args,
                    detector_dir=args.detector_dir,
                    detector_output=detector_output,
                    evaluation_dir=evaluation_output,
                    output_dir=gate_output,
                ),
                gpu=args.defense_gpu,
                log_path=logs / "gate.log",
                stage="gate",
            )

        if not gate_complete(gate_output / "result.json"):
            raise RuntimeError(f"Gate output failed validation: {gate_output}")
        result = read_result(dataset_run)
        clean_gated_result = args.clean_gated_root / dataset_slug / "result.json"
        if clean_gated_result.is_file():
            clean_payload = load_json(clean_gated_result)
            result["clean_un_gated"] = result["clean"]
            result["clean_gated"] = float(clean_payload["clean_gated"])
            result["clean_flag_rate"] = float(clean_payload["clean_flag_rate"])
            result["clean"] = result["clean_gated"]
        original_clean = (
            load_json(evaluation_output / "clean_original_reference.json")["accuracy"]
            if not args.skip_base_clean
            else float("nan")
        )
        row = {
            "category": category,
            "dataset": dataset_name,
            "slug": dataset_slug,
            "sample_count": args.sample_count,
            "original_clean_reference": float(original_clean),
            **result,
            "stage_times": stage_times,
            "run_dir": str(dataset_run),
            "attack_summary": str(attack_dir / "pgd_generation_summary.json"),
        }
        if result.get("auc_ci95") and len(result["auc_ci95"]) == 2:
            row["auc_ci95_low"] = float(result["auc_ci95"][0])
            row["auc_ci95_high"] = float(result["auc_ci95"][1])
        (dataset_run / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(
            f"    result: Clean={100.0 * row['clean']:.2f}% | "
            f"Raw Rob={100.0 * row['raw_robust']:.2f}% | "
            f"Corrected Rob={100.0 * row['corrected_robust']:.2f}% | "
            f"Gated Rob={100.0 * row['gated_robust']:.2f}% | AUC={row['auc']:.4f}",
            flush=True,
        )
        write_aggregate(
            args=args,
            selected=selected,
            rows=rows,
            edited_sha=edited_sha,
            started=started,
        )

    # Re-read all completed rows so a resumed run always writes a complete table.
    all_rows = []
    for category, dataset_name in selected:
        result_path = args.run_root / slug(category, dataset_name) / "result.json"
        if result_path.is_file():
            all_rows.append(load_json(result_path))
    if len(all_rows) == len(selected):
        write_aggregate(
            args=args,
            selected=selected,
            rows=all_rows,
            edited_sha=edited_sha,
            started=started,
        )
    print("\n[table1] all requested datasets complete", flush=True)


if __name__ == "__main__":
    main()

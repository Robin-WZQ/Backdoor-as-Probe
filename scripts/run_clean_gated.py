#!/usr/bin/env python3

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


def select_datasets(value: str) -> list[tuple[str, str]]:
    """Resolve the requested subset using the same names as the Table-1 runner."""

    if value.strip().lower() == "all":
        return list(DATASETS)
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    selected = [
        pair
        for pair in DATASETS
        if pair[1].lower() in requested or slug(*pair) in requested
    ]
    known = {pair[1].lower() for pair in selected} | {slug(*pair) for pair in selected}
    missing = requested - known
    if missing:
        raise ValueError(f"Unknown datasets: {sorted(missing)}")
    return selected


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    runtime = repo_root / "runtime"
    artifact = runtime / "artifacts/output_low_energy"
    table1 = runtime / "runs/table1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo_root)
    parser.add_argument(
        "--clean_root",
        type=Path,
        default=repo_root / "data/tables/clean",
    )
    parser.add_argument("--table1_root", type=Path, default=table1)
    parser.add_argument("--edited_model_dir", type=Path, default=artifact / "edited_model")
    parser.add_argument("--clean_features_path", type=Path, default=artifact / "clean_features_layers_6_token_0.pt")
    parser.add_argument(
        "--direction_artifact",
        type=Path,
        default=artifact / "global_correction_directions.pt",
    )
    parser.add_argument(
        "--threshold_path",
        type=Path,
        default=artifact / "target_response_q95/threshold.pt",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=runtime / "runs/table1_clean_gated",
    )
    parser.add_argument("--sample_count", type=int, default=1000)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names (default: all 16), e.g. ImageNet,Flowers102",
    )
    parser.add_argument("--mlp_layer", type=int, default=6)
    parser.add_argument("--token_index", type=int, default=0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--benign_direction_weight", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--correction_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument("--gpu", default="0", help="Physical GPU for correction/evaluation")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_count(path: Path) -> int:
    return len(csv_rows(path)) if path.is_file() else 0


def image_count(path: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not path.is_dir():
        return 0
    return sum(
        item.is_file() and item.suffix.lower() in extensions
        for item in path.iterdir()
    )


def model_sha256(model_dir: Path) -> str:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    raise FileNotFoundError(f"No model weight file under {model_dir}")


def run_stage(
    *,
    args: argparse.Namespace,
    command: list[object],
    log_path: Path,
    stage: str,
) -> float:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    repo = resolve(args.repo)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(str(item) for item in command)
    print(f"    [{stage}] GPU {args.gpu}: {rendered}", flush=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {rendered}\n")
        handle.flush()
        subprocess.run(
            [str(item) for item in command],
            cwd=repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.perf_counter() - started
    print(f"    [{stage}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def correction_complete(path: Path, count: int) -> bool:
    summary_path = path / "correction_summary.json"
    return (
        summary_path.is_file()
        and (path / "correction.csv").is_file()
        and image_count(path / "images") == count
        and csv_count(path / "correction.csv") == count
        and load_json(summary_path).get("num_images") == count
    )


def accuracy_complete(path: Path, count: int) -> bool:
    return (
        path.is_file()
        and load_json(path).get("total") == count
        and csv_count(path.with_suffix(".csv")) == count
    )


def aligned(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        stem = Path(row["image_name"]).stem
        if stem in output:
            raise ValueError(f"Duplicate image stem: {stem}")
        output[stem] = row
    return output


def gated_accuracy(
    detector_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    corrected_rows: list[dict[str, str]],
    threshold: float,
) -> dict[str, Any]:
    detector = aligned(detector_rows)
    raw = aligned(raw_rows)
    corrected = aligned(corrected_rows)
    if set(detector) != set(raw) or set(detector) != set(corrected):
        raise ValueError(
            "Clean detector/raw/corrected outputs do not contain the same image stems"
        )
    total = len(detector)
    correct = raw_count = corrected_count = 0
    correct_raw = correct_corrected = 0
    for stem, score_row in detector.items():
        use_corrected = float(score_row["score"]) >= threshold
        selected = corrected[stem] if use_corrected else raw[stem]
        is_correct = int(selected["correct"])
        correct += is_correct
        if use_corrected:
            corrected_count += 1
            correct_corrected += is_correct
        else:
            raw_count += 1
            correct_raw += is_correct
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
        "corrected_branch_count": corrected_count,
        "raw_branch_count": raw_count,
        "correct_from_corrected_branch": correct_corrected,
        "correct_from_raw_branch": correct_raw,
        "clean_flag_rate": corrected_count / total if total else 0.0,
    }


def correction_command(args: argparse.Namespace, image_dir: Path, output_dir: Path) -> list[str]:
    command: list[object] = [
        args.python,
        "-m",
        "BaP.correction.current",
        "--edited_model_dir",
        resolve(args.edited_model_dir),
        "--image_dir",
        image_dir,
        "--clean_features_path",
        resolve(args.clean_features_path),
        "--global_direction_artifact",
        resolve(args.direction_artifact),
        "--output_dir",
        output_dir,
        "--device",
        "cuda:0",
        "--image_size",
        args.image_size,
        "--batch_size",
        args.correction_batch_size,
        "--max_images",
        args.sample_count,
        "--mlp_layer",
        args.mlp_layer,
        "--token_index",
        args.token_index,
        "--residual_weight",
        args.residual_weight,
        "--benign_direction_weight",
        args.benign_direction_weight,
    ]
    if args.local_files_only:
        command.append("--local_files_only")
    return command


def accuracy_command(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    image_dir: Path,
    output_json: Path,
    dataset_name: str,
) -> list[str]:
    command: list[object] = [
        args.python,
        "-m",
        "BaP.evaluation.accuracy",
        "--image_dir",
        image_dir,
        "--labels_csv",
        dataset_root / "labels.csv",
        "--classes_json",
        dataset_root / "classes.json",
        "--dataset_name",
        dataset_name,
        "--model_id",
        resolve(args.edited_model_dir),
        "--device",
        "cuda:0",
        "--batch_size",
        args.eval_batch_size,
        "--num_workers",
        args.num_workers,
        "--image_preprocess",
        "fixed_224",
        "--limit",
        args.sample_count,
        "--output_json",
        output_json,
        "--output_csv",
        output_json.with_suffix(".csv"),
    ]
    if args.local_files_only:
        command.append("--local_files_only")
    return command


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# Table 1 — Clean gated evaluation",
        "",
        "本报告把 1000 张 clean 样本完整送入 BaP gate：超过 detector 阈值的样本执行 canonical E2R1，未超过阈值的样本直接使用 Edited CLIP；两个分支均由 Edited CLIP 分类。",
        "",
        "## Protocol",
        "",
        f"- Edited model: `{config['edited_model_dir']}`",
        f"- Detector threshold: `{config['threshold']:.12f}` (global benign q95)",
        "- Clean detector scores: the same target-response detector and fixed 224x224 defense preprocessing as Table 1.",
        "- Correction: canonical E2R1 (2-step feature-drift escape + 1-step benign-only R1 repair), epsilon 4/255.",
        "- Final classification in both branches: Edited CLIP.",
        f"- Samples: {config['sample_count']} per dataset, {len(rows)} datasets.",
        f"- Environment: `backdoor_defense`; physical GPU {config['gpu']} for correction/evaluation.",
        "",
        "## Results",
        "",
        "`Clean (processor)` is the earlier Table-1 clean number; `Clean (fixed raw)` is the raw branch under the defense input geometry; `Clean (gated)` is the requested complete-gate metric.",
        "",
        "| Category | Dataset | Clean (processor) | Clean (fixed raw) | Always corrected | Clean (gated) | Clean flag rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['dataset']} | {pct(row['clean_processor'])} | "
            f"{pct(row['clean_fixed_raw'])} | {pct(row['clean_always_corrected'])} | "
            f"{pct(row['clean_gated'])} | {pct(row['clean_flag_rate'])} |"
        )
    keys = [
        "clean_processor",
        "clean_fixed_raw",
        "clean_always_corrected",
        "clean_gated",
        "clean_flag_rate",
    ]
    macro = {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}
    lines.extend(
        [
            "",
            "## Macro average",
            "",
            f"- Clean (processor): **{pct(macro['clean_processor'])}**",
            f"- Clean (fixed raw): **{pct(macro['clean_fixed_raw'])}**",
            f"- Always corrected: **{pct(macro['clean_always_corrected'])}**",
            f"- Clean (gated): **{pct(macro['clean_gated'])}**",
            f"- Clean flag rate: **{pct(macro['clean_flag_rate'])}**",
            "",
            "## Output audit",
            "",
            "- Every dataset has 1000 detector rows, 1000 raw predictions, 1000 corrected predictions, and 1000 gated decisions.",
            "- `Clean (gated)` is not the un-gated `Clean` column from the attack report; it includes correction for detector-flagged clean samples.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.repo = resolve(args.repo)
    args.clean_root = resolve(args.clean_root)
    args.table1_root = resolve(args.table1_root)
    args.edited_model_dir = resolve(args.edited_model_dir)
    args.clean_features_path = resolve(args.clean_features_path)
    args.direction_artifact = resolve(args.direction_artifact)
    args.threshold_path = resolve(args.threshold_path)
    args.output_root = resolve(args.output_root)
    selected_datasets = select_datasets(args.datasets)
    if not selected_datasets:
        raise ValueError("At least one dataset must be selected")
    for required in (
        args.clean_root,
        args.table1_root,
        args.edited_model_dir,
        args.clean_features_path,
        args.direction_artifact,
        args.threshold_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    import torch

    threshold = float(torch.load(args.threshold_path, map_location="cpu", weights_only=True))
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "script": str(Path(__file__).resolve()),
        "repo": str(args.repo),
        "edited_model_dir": str(args.edited_model_dir),
        "edited_model_sha256": model_sha256(args.edited_model_dir),
        "clean_root": str(args.clean_root),
        "table1_root": str(args.table1_root),
        "clean_features_path": str(args.clean_features_path),
        "direction_artifact": str(args.direction_artifact),
        "threshold_path": str(args.threshold_path),
        "threshold": threshold,
        "threshold_quantile": 0.95,
        "sample_count": args.sample_count,
        "datasets": [f"{category}/{name}" for category, name in selected_datasets],
        "gpu": args.gpu,
        "environment": "backdoor_defense",
        "local_files_only": args.local_files_only,
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for index, (category, dataset_name) in enumerate(selected_datasets, start=1):
        dataset_root = args.clean_root / category / dataset_name
        table1_dataset = args.table1_root / slug(category, dataset_name)
        detector_csv = table1_dataset / "target_response_auc/clean_scores.csv"
        if not dataset_root.is_dir() or not detector_csv.is_file():
            raise FileNotFoundError(f"Missing clean dataset or detector scores for {dataset_name}")
        output = args.output_root / slug(category, dataset_name)
        correction_output = output / "correction_clean"
        evaluation_output = output / "evaluation"
        logs = output / "logs"
        output.mkdir(parents=True, exist_ok=True)
        evaluation_output.mkdir(parents=True, exist_ok=True)
        print(f"\n[clean-gated] {index}/{len(selected_datasets)} {category}/{dataset_name}", flush=True)
        stage_times: dict[str, float] = {}

        if not correction_complete(correction_output, args.sample_count):
            stage_times["correct_clean"] = run_stage(
                args=args,
                command=correction_command(args, dataset_root / "images", correction_output),
                log_path=logs / "correct_clean.log",
                stage="correct_clean",
            )
        else:
            print("    [correct_clean] valid result already exists; reusing", flush=True)
        if not correction_complete(correction_output, args.sample_count):
            raise RuntimeError(f"Correction output failed validation: {correction_output}")

        raw_json = evaluation_output / "clean_fixed_raw_edited.json"
        corrected_json = evaluation_output / "clean_corrected_edited.json"
        if not accuracy_complete(raw_json, args.sample_count):
            stage_times["eval_clean_fixed_raw"] = run_stage(
                args=args,
                command=accuracy_command(
                    args,
                    dataset_root=dataset_root,
                    image_dir=dataset_root / "images",
                    output_json=raw_json,
                    dataset_name=dataset_name,
                ),
                log_path=logs / "eval_clean_fixed_raw.log",
                stage="eval_clean_fixed_raw",
            )
        else:
            print("    [eval_clean_fixed_raw] valid result already exists; reusing", flush=True)
        if not accuracy_complete(corrected_json, args.sample_count):
            stage_times["eval_clean_corrected"] = run_stage(
                args=args,
                command=accuracy_command(
                    args,
                    dataset_root=dataset_root,
                    image_dir=correction_output / "images",
                    output_json=corrected_json,
                    dataset_name=dataset_name,
                ),
                log_path=logs / "eval_clean_corrected.log",
                stage="eval_clean_corrected",
            )
        else:
            print("    [eval_clean_corrected] valid result already exists; reusing", flush=True)

        detector_rows = csv_rows(detector_csv)
        if len(detector_rows) != args.sample_count:
            raise ValueError(f"Expected {args.sample_count} clean detector rows, found {len(detector_rows)}")
        saved_thresholds = {float(row["threshold"]) for row in detector_rows}
        if len(saved_thresholds) != 1 or abs(next(iter(saved_thresholds)) - threshold) > 1e-9:
            raise ValueError(f"Detector threshold mismatch for {dataset_name}: {saved_thresholds}")
        raw_rows = csv_rows(raw_json.with_suffix(".csv"))
        corrected_rows = csv_rows(corrected_json.with_suffix(".csv"))
        gated = gated_accuracy(detector_rows, raw_rows, corrected_rows, threshold)
        raw_result = load_json(raw_json)
        corrected_result = load_json(corrected_json)
        processor_result = load_json(table1_dataset / "evaluation/clean_edited.json")
        row = {
            "category": category,
            "dataset": dataset_name,
            "slug": slug(category, dataset_name),
            "sample_count": args.sample_count,
            "clean_processor": float(processor_result["accuracy"]),
            "clean_fixed_raw": float(raw_result["accuracy"]),
            "clean_always_corrected": float(corrected_result["accuracy"]),
            "clean_gated": float(gated["accuracy"]),
            "clean_flag_rate": float(gated["clean_flag_rate"]),
            "clean_gated_correct": int(gated["correct"]),
            "clean_gated_corrected_branch_count": int(gated["corrected_branch_count"]),
            "clean_gated_raw_branch_count": int(gated["raw_branch_count"]),
            "clean_gated_correct_from_corrected": int(gated["correct_from_corrected_branch"]),
            "clean_gated_correct_from_raw": int(gated["correct_from_raw_branch"]),
            "threshold": threshold,
            "stage_times": stage_times,
            "run_dir": str(output),
        }
        (output / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(
            f"    result: processor={100*row['clean_processor']:.2f}% | "
            f"fixed_raw={100*row['clean_fixed_raw']:.2f}% | "
            f"always_corrected={100*row['clean_always_corrected']:.2f}% | "
            f"gated={100*row['clean_gated']:.2f}% | "
            f"flag_rate={100*row['clean_flag_rate']:.2f}%",
            flush=True,
        )

    # Re-read all per-dataset files so a resumed run always writes a complete report.
    rows = [
        load_json(args.output_root / slug(category, name) / "result.json")
        for category, name in selected_datasets
    ]
    write_csv(args.output_root / "clean_gated_results.csv", rows)
    write_report(args.output_root / "REPORT.md", rows, config)
    print(f"[clean-gated] report written: {args.output_root / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

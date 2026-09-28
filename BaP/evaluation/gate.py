from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def read_detector_csv(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            rows[row["image_name"]] = {
                "score": float(row["score"]),
                "saved_threshold": float(row["threshold"]),
                "saved_is_suspicious": int(row["is_suspicious"]),
            }
    if not rows:
        raise ValueError(f"No detector rows found: {path}")
    return rows


def read_prediction_csv(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {row["image_name"]: row for row in csv.DictReader(f)}
    if not rows:
        raise ValueError(f"No prediction rows found: {path}")
    return rows


def load_saved_threshold(
    threshold_path: Path,
    benign_scores: dict[str, dict],
    attack_scores: dict[str, dict],
) -> float:
    threshold = float(
        torch.load(threshold_path, map_location="cpu", weights_only=True)
    )
    csv_thresholds = {
        row["saved_threshold"]
        for scores in (benign_scores, attack_scores)
        for row in scores.values()
    }
    if len(csv_thresholds) != 1:
        raise ValueError(
            "Expected one saved threshold across benign and attack detector CSVs, "
            f"found {sorted(csv_thresholds)}"
        )
    csv_threshold = csv_thresholds.pop()
    if abs(csv_threshold - threshold) > 1e-9:
        raise ValueError(
            f"Detector CSV threshold {csv_threshold} does not match model artifact "
            f"threshold {threshold} from {threshold_path}"
        )
    return threshold


def detection_stats(benign_scores: dict[str, dict], attack_scores: dict[str, dict], threshold: float) -> dict:
    tp = sum(1 for row in attack_scores.values() if row["score"] >= threshold)
    fp = sum(1 for row in benign_scores.values() if row["score"] >= threshold)
    fn = len(attack_scores) - tp
    tn = len(benign_scores) - fp
    tpr = tp / len(attack_scores)
    fpr = fp / len(benign_scores)
    tnr = tn / len(benign_scores)
    return {
        "threshold": threshold,
        "num_benign": len(benign_scores),
        "num_attack": len(attack_scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "youden_j": tpr - fpr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "distance_to_top_left": ((1.0 - tpr) ** 2 + fpr**2) ** 0.5,
    }


def threshold_candidates(benign_scores: dict[str, dict], attack_scores: dict[str, dict]) -> list[float]:
    scores = [row["score"] for row in benign_scores.values()]
    scores.extend(row["score"] for row in attack_scores.values())
    unique_scores = sorted(set(scores), reverse=True)
    eps = 1e-12
    return [max(unique_scores) + eps, *unique_scores, min(unique_scores) - eps]


def select_youden_threshold(benign_scores: dict[str, dict], attack_scores: dict[str, dict]) -> tuple[float, dict]:
    best = None
    best_stats = None
    for threshold in threshold_candidates(benign_scores, attack_scores):
        stats = detection_stats(benign_scores, attack_scores, threshold)
        candidate = (
            stats["youden_j"],
            -stats["distance_to_top_left"],
            -stats["fpr"],
            threshold,
        )
        if best is None or candidate > best:
            best = candidate
            best_stats = stats
    return best_stats["threshold"], best_stats


def gated_rows(
    attack_scores: dict[str, dict],
    raw_predictions: dict[str, dict],
    corrected_predictions: dict[str, dict],
    threshold: float,
) -> tuple[list[dict], dict]:
    missing = [
        name
        for name in attack_scores
        if name not in raw_predictions or name not in corrected_predictions
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"{len(missing)} attack images missing predictions: {preview}{suffix}")

    rows = []
    correct = 0
    corrected_count = 0
    raw_count = 0
    correct_from_corrected = 0
    correct_from_raw = 0

    for name in sorted(attack_scores):
        score = attack_scores[name]["score"]
        detected = score >= threshold
        selected = corrected_predictions[name] if detected else raw_predictions[name]
        selected_correct = int(selected["correct"])
        correct += selected_correct

        if detected:
            corrected_count += 1
            correct_from_corrected += selected_correct
        else:
            raw_count += 1
            correct_from_raw += selected_correct

        rows.append(
            {
                "image_name": name,
                "detector_score": f"{score:.8f}",
                "gate_threshold": f"{threshold:.8f}",
                "is_detected_adv": int(detected),
                "selected_branch": "bap_corrected" if detected else "raw",
                "label_idx": selected["label_idx"],
                "label": selected["label"],
                "pred_idx": selected["pred_idx"],
                "pred": selected["pred"],
                "correct": selected_correct,
                "raw_correct": int(raw_predictions[name]["correct"]),
                "bap_corrected_correct": int(corrected_predictions[name]["correct"]),
            }
        )

    total = len(rows)
    return rows, {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
        "corrected_count": corrected_count,
        "raw_count": raw_count,
        "correct_from_corrected_branch": correct_from_corrected,
        "correct_from_raw_branch": correct_from_raw,
    }


def select_class_accuracy_threshold(
    benign_scores: dict[str, dict],
    attack_scores: dict[str, dict],
    raw_predictions: dict[str, dict],
    corrected_predictions: dict[str, dict],
) -> tuple[float, dict, dict]:
    best = None
    best_stats = None
    best_accuracy = None
    for threshold in threshold_candidates(benign_scores, attack_scores):
        _, accuracy = gated_rows(attack_scores, raw_predictions, corrected_predictions, threshold)
        stats = detection_stats(benign_scores, attack_scores, threshold)
        candidate = (
            accuracy["correct"],
            stats["youden_j"],
            -stats["fpr"],
            threshold,
        )
        if best is None or candidate > best:
            best = candidate
            best_stats = stats
            best_accuracy = accuracy
    return best_stats["threshold"], best_stats, best_accuracy


def baseline_accuracy(predictions: dict[str, dict]) -> dict:
    correct = sum(int(row["correct"]) for row in predictions.values())
    total = len(predictions)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, result: dict) -> None:
    det = result["detection"]
    acc = result["gated_accuracy_on_attack"]
    baselines = result["baselines"]
    lines = [
        "# BaP gated accuracy",
        "",
        "## Threshold",
        "",
        f"- Threshold mode: `{result['threshold_mode']}`",
        f"- Gate threshold: `{result['gate_threshold']:.8f}`",
        f"- Detection rule: adversarial if `score >= threshold`",
        f"- Detection TPR: `{det['tpr'] * 100:.2f}%`",
        f"- Detection FPR: `{det['fpr'] * 100:.2f}%`",
        f"- Youden J: `{det['youden_j']:.6f}`",
        "",
        "## Accuracy",
        "",
        "| Method | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
        f"| Raw PGD 1/255 | {baselines['raw_pgd']['correct']} | {baselines['raw_pgd']['total']} | {baselines['raw_pgd']['accuracy_percent']:.2f}% |",
        f"| BaP correction on all samples | {baselines['bap_corrected']['correct']} | {baselines['bap_corrected']['total']} | {baselines['bap_corrected']['accuracy_percent']:.2f}% |",
        f"| BaP gated correction | {acc['correct']} | {acc['total']} | {acc['accuracy_percent']:.2f}% |",
        "",
        "## Gate Counts",
        "",
        "| Branch | Count | Correct |",
        "| --- | ---: | ---: |",
        f"| BaP corrected | {acc['corrected_count']} | {acc['correct_from_corrected_branch']} |",
        f"| Raw | {acc['raw_count']} | {acc['correct_from_raw_branch']} |",
        "",
        "## Inputs",
        "",
    ]
    for key, value in result["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate probe-gated BaP correction from detector scores and prediction CSVs."
    )
    parser.add_argument(
        "--benign_detector_csv",
        required=True,
    )
    parser.add_argument(
        "--attack_detector_csv",
        required=True,
    )
    parser.add_argument(
        "--raw_predictions_csv",
        required=True,
    )
    parser.add_argument(
        "--corrected_predictions_csv",
        required=True,
    )
    parser.add_argument(
        "--threshold_mode",
        choices=["fixed", "youden", "class_accuracy", "saved", "manual"],
        default="saved",
        help="fixed uses the configured threshold; youden and class_accuracy are diagnostic modes.",
    )
    parser.add_argument("--fixed_threshold", type=float, default=None)
    parser.add_argument("--manual_threshold", type=float, default=None)
    parser.add_argument(
        "--threshold_path",
        default=None,
        help="Model artifact threshold.pt; required when threshold_mode is saved.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        required=True,
    )
    parser.add_argument(
        "--output_md",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benign_scores = read_detector_csv(Path(args.benign_detector_csv))
    attack_scores = read_detector_csv(Path(args.attack_detector_csv))
    raw_predictions = read_prediction_csv(Path(args.raw_predictions_csv))
    corrected_predictions = read_prediction_csv(Path(args.corrected_predictions_csv))

    if args.threshold_mode == "fixed":
        if args.fixed_threshold is None:
            raise ValueError("--fixed_threshold is required when --threshold_mode fixed")
        threshold = args.fixed_threshold
        det_stats = detection_stats(benign_scores, attack_scores, threshold)
    elif args.threshold_mode == "youden":
        threshold, det_stats = select_youden_threshold(benign_scores, attack_scores)
    elif args.threshold_mode == "class_accuracy":
        threshold, det_stats, _ = select_class_accuracy_threshold(
            benign_scores,
            attack_scores,
            raw_predictions,
            corrected_predictions,
        )
    elif args.threshold_mode == "saved":
        if args.threshold_path is None:
            raise ValueError("--threshold_path is required when --threshold_mode saved")
        threshold = load_saved_threshold(
            Path(args.threshold_path), benign_scores, attack_scores
        )
        det_stats = detection_stats(benign_scores, attack_scores, threshold)
    else:
        if args.manual_threshold is None:
            raise ValueError("--manual_threshold is required when --threshold_mode manual")
        threshold = args.manual_threshold
        det_stats = detection_stats(benign_scores, attack_scores, threshold)

    rows, gated_accuracy = gated_rows(attack_scores, raw_predictions, corrected_predictions, threshold)
    result = {
        "threshold_mode": args.threshold_mode,
        "gate_threshold": threshold,
        "detection": det_stats,
        "gated_accuracy_on_attack": gated_accuracy,
        "baselines": {
            "raw_pgd": baseline_accuracy(raw_predictions),
            "bap_corrected": baseline_accuracy(corrected_predictions),
        },
        "inputs": {
            "benign_detector_csv": args.benign_detector_csv,
            "attack_detector_csv": args.attack_detector_csv,
            "raw_predictions_csv": args.raw_predictions_csv,
            "corrected_predictions_csv": args.corrected_predictions_csv,
            "threshold_path": args.threshold_path,
        },
        "outputs": {
            "output_json": args.output_json,
            "output_csv": args.output_csv,
            "output_md": args.output_md,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    write_report(Path(args.output_md), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

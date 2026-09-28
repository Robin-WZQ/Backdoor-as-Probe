from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_scores(csv_path: Path) -> list[float]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [float(row["score"]) for row in reader]


def roc_curve(labels: list[int], scores: list[float]):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos_total = sum(labels)
    neg_total = len(labels) - pos_total
    if pos_total == 0 or neg_total == 0:
        raise ValueError("ROC requires both positive and negative samples.")

    tprs = [0.0]
    fprs = [0.0]
    tp = 0
    fp = 0
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            tprs.append(tp / pos_total)
            fprs.append(fp / neg_total)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    tprs.append(tp / pos_total)
    fprs.append(fp / neg_total)
    return fprs, tprs


def auc_from_roc(fprs: list[float], tprs: list[float]) -> float:
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2.0
    return auc


def render_svg(comparisons, output_path: Path):
    width, height = 840, 640
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 30, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = ["#d1495b", "#00798c", "#edae49", "#30638e", "#6a4c93"]

    def px(x):
        return margin_left + x * plot_w

    def py(y):
        return height - margin_bottom - y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="20" font-family="Arial">BaP Probe-Response ROC Curves</text>',
    ]

    # Grid
    for tick in range(6):
        value = tick / 5
        x = px(value)
        y = py(value)
        parts.append(f'<line x1="{x}" y1="{py(0)}" x2="{x}" y2="{py(1)}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<line x1="{px(0)}" y1="{y}" x2="{px(1)}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x}" y="{height - margin_bottom + 20}" text-anchor="middle" font-size="12" font-family="Arial">{value:.1f}</text>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial">{value:.1f}</text>')

    # Axes
    parts.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(0)}" stroke="black" stroke-width="2"/>')
    parts.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(0)}" y2="{py(1)}" stroke="black" stroke-width="2"/>')
    parts.append(f'<text x="{width/2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">False Positive Rate</text>')
    parts.append(
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {height/2})">True Positive Rate</text>'
    )

    # Diagonal
    parts.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="6,4"/>')

    # Curves
    legend_y = margin_top + 20
    legend_x = width - 250
    for idx, item in enumerate(comparisons):
        color = colors[idx % len(colors)]
        points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(item["fprs"], item["tprs"]))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        parts.append(f'<line x1="{legend_x}" y1="{legend_y + idx*24}" x2="{legend_x + 24}" y2="{legend_y + idx*24}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 32}" y="{legend_y + idx*24 + 4}" font-size="13" font-family="Arial">{item["attack_name"]} (AUC={item["auc"]:.4f})</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def compare_one(benign_csv: Path, attack_csv: Path):
    benign_scores = read_scores(benign_csv)
    attack_scores = read_scores(attack_csv)
    scores = benign_scores + attack_scores
    labels = [0] * len(benign_scores) + [1] * len(attack_scores)
    fprs, tprs = roc_curve(labels, scores)
    auc = auc_from_roc(fprs, tprs)
    return {
        "attack_name": attack_csv.stem.replace("suspicious_samples_", ""),
        "auc": auc,
        "num_benign": len(benign_scores),
        "num_attack": len(attack_scores),
        "fprs": fprs,
        "tprs": tprs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC/AUC from BaP probe-response score CSVs.")
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--benign_csv", default=None, help="Optional explicit benign csv path.")
    parser.add_argument("--attack_csvs", nargs="*", default=None, help="Optional explicit attack csv paths.")
    parser.add_argument("--output_svg", default="output/score_auc.svg")
    parser.add_argument("--output_json", default="output/score_auc_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    benign_csv = Path(args.benign_csv) if args.benign_csv else input_dir / "suspicious_samples_Benign.csv"
    if not benign_csv.exists():
        raise FileNotFoundError(f"Benign CSV not found: {benign_csv}")

    if args.attack_csvs:
        attack_csvs = [Path(path) for path in args.attack_csvs]
    else:
        attack_csvs = sorted(
            p
            for p in input_dir.glob("suspicious_samples_*.csv")
            if p.name != benign_csv.name and p.is_file()
        )
    if not attack_csvs:
        raise ValueError("No attack CSVs found.")

    comparisons = [compare_one(benign_csv, attack_csv) for attack_csv in attack_csvs]

    output_svg = Path(args.output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    render_svg(comparisons, output_svg)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "benign_csv": str(benign_csv),
        "attack_csvs": [str(path) for path in attack_csvs],
        "comparisons": [
            {
                "attack_name": item["attack_name"],
                "auc": item["auc"],
                "num_benign": item["num_benign"],
                "num_attack": item["num_attack"],
            }
            for item in comparisons
        ],
        "output_svg": str(output_svg),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

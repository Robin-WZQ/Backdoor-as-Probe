#!/usr/bin/env python3
"""Build a BaP benign residual-direction artifact from calibration images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BaP.correction.current import (
    BenignDirectionDataset,
    build_direction_bank,
    build_projections,
    load_clip_model,
)
from BaP import enforce_gpu_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited_model_dir", required=True)
    parser.add_argument("--benign_dir", required=True)
    parser.add_argument("--clean_features_path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlp_layers", type=lambda value: [int(x) for x in value.split(",")], required=True)
    parser.add_argument("--token_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--svd_eps", type=float, default=1e-6)
    parser.add_argument("--direction_eps", type=float, default=1e-8)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_gpu_policy(args.device)
    loaded = load_clip_model(
        args.edited_model_dir,
        args.device,
        local_files_only=args.local_files_only,
    )
    projections, ranks = build_projections(args, loaded.device)
    dataset = BenignDirectionDataset(
        args.benign_dir, args.image_size, args.max_images
    )
    directions = build_direction_bank(loaded, dataset, projections, args)
    payload = {
        "version": 2,
        "method": "one pooled mixed-domain BaP benign direction bank",
        "domain_policy": "no dataset IDs and no test-set direction fitting",
        "edited_model_dir": str(Path(args.edited_model_dir).resolve()),
        "benign_dir": str(Path(args.benign_dir).resolve()),
        "clean_features_path": str(Path(args.clean_features_path).resolve()),
        "mlp_layers": args.mlp_layers,
        "token_index": args.token_index,
        "count": len(dataset),
        "projection_ranks": ranks,
        "benign": {layer: value.detach().cpu() for layer, value in directions.benign.items()},
        "benign_concentration": directions.benign_concentration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        key: value
        for key, value in payload.items()
        if key != "benign"
    }
    summary["output"] = str(args.output.resolve())
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from ..core.clip import encode_text_features, load_clip_model, set_seed
from .. import enforce_gpu_policy
from ..core.protocol import (
    ATTACK_IMPLEMENTATIONS,
    ATTACK_GENERATOR,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    IMAGE_SERIALIZATION,
    TEXT_BATCH_SIZE,
    ZERO_SHOT_PROMPT,
)
from ..core.datasets import LabeledImageDirectory
from .methods import ZeroShotCLIPClassifier, pgd_attack


def local_model_sha256(model_id: str) -> str | None:
    model_dir = Path(model_id)
    if not model_dir.is_dir():
        return None
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return None


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    # CSR uses ToPILImage on float tensors, which scales by 255 and truncates to uint8.
    to_pil_image(tensor.detach().clamp(0.0, 1.0).cpu()).save(path)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=-1)
    correct = preds.eq(labels)
    return preds, correct


def parse_args():
    parser = argparse.ArgumentParser(description="Generate attacks against a BaP probe model.")
    parser.add_argument("--attack", choices=("pgd", "autoattack_apgd"), default="pgd")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--classes_json", default=None)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    enforce_gpu_policy(args.device)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = "pgd" if args.attack == "pgd" else "autoattack_apgd"
    summary_json = Path(args.summary_json) if args.summary_json else output_dir / f"{output_stem}_generation_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / f"{output_stem}_generation_samples.csv"

    dataset = LabeledImageDirectory(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        dataset_name=args.dataset_name,
        classes_json=args.classes_json,
        image_size=args.image_size,
        limit=args.limit,
    )
    loaded = load_clip_model(
        args.model_id, args.device, local_files_only=args.local_files_only
    )
    text_features = encode_text_features(
        loaded, dataset.class_names, batch_size=TEXT_BATCH_SIZE
    ).to(loaded.device)
    classifier = ZeroShotCLIPClassifier(loaded=loaded, text_features=text_features)
    alpha = None
    if args.attack == "pgd":
        alpha = args.alpha if args.alpha is not None else (args.epsilon / args.steps) * args.alpha_scale

    rows = []
    clean_correct = 0
    adv_correct = 0
    total = 0
    linf_values = []

    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"Generating {args.attack}"):
        images, labels, names, source_names = dataset.batch(start, args.batch_size)
        labels = labels.to(loaded.device)
        images = images.to(loaded.device)

        output_paths = [output_dir / f"{Path(name).stem}.png" for name in names]
        if args.skip_existing and all(path.exists() for path in output_paths):
            continue

        with torch.no_grad():
            clean_logits = classifier.logits(images)
            clean_preds, clean_ok = accuracy_from_logits(clean_logits, labels)

        if args.attack == "pgd":
            adv = pgd_attack(
                classifier=classifier,
                images=images,
                labels=labels,
                epsilon=args.epsilon,
                steps=args.steps,
                alpha=alpha,
            )


        with torch.no_grad():
            adv_logits = classifier.logits(adv)
            adv_preds, adv_ok = accuracy_from_logits(adv_logits, labels)

        linf = (adv - images).flatten(1).abs().max(dim=1).values
        clean_correct += int(clean_ok.sum().item())
        adv_correct += int(adv_ok.sum().item())
        total += int(labels.numel())
        linf_values.extend(float(x) for x in linf.detach().cpu().tolist())

        for i, (name, source_name) in enumerate(zip(names, source_names)):
            save_tensor_image(adv[i], output_paths[i])
            rows.append(
                {
                    "image_name": f"{Path(name).stem}.png",
                    "source_image": source_name,
                    "label_idx": int(labels[i].item()),
                    "label": dataset.class_names[int(labels[i].item())],
                    "clean_pred_idx": int(clean_preds[i].item()),
                    "clean_pred": dataset.class_names[int(clean_preds[i].item())],
                    "clean_correct": int(clean_ok[i].item()),
                    "adv_pred_idx": int(adv_preds[i].item()),
                    "adv_pred": dataset.class_names[int(adv_preds[i].item())],
                    "adv_correct": int(adv_ok[i].item()),
                    "linf_to_clean": f"{linf[i].item():.8f}",
                }
            )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    linf_t = torch.tensor(linf_values)
    summary = {
        "model_id": args.model_id,
        "attack_model_sha256": local_model_sha256(args.model_id),
        "dataset_name": args.dataset_name,
        "attack": args.attack,
        "attack_implementation": ATTACK_IMPLEMENTATIONS[args.attack],
        "autoattack_scope": "APGD-CE only" if args.attack == "autoattack_apgd" else None,
        "image_dir": args.image_dir,
        "labels_csv": args.labels_csv,
        "output_dir": str(output_dir),
        "total": total,
        "epsilon": args.epsilon,
        "steps": args.steps,
        "alpha": alpha,
        "alpha_scale": args.alpha_scale,
        "prompt": ZERO_SHOT_PROMPT,
        "text_batch_size": TEXT_BATCH_SIZE,
        "input_preprocess": "bicubic resize to 224x224, center crop 224, float tensor in [0,1]",
        "storage": "8-bit PNG",
        "serialization": IMAGE_SERIALIZATION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_attack_generator": ATTACK_GENERATOR,
        "seed": args.seed,
        "logical_device": args.device,
        "visible_physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "clean_correct_on_attack_model": clean_correct,
        "clean_accuracy_on_attack_model": clean_correct / total if total else 0.0,
        "clean_accuracy_percent_on_attack_model": 100.0 * clean_correct / total if total else 0.0,
        "adv_correct_on_attack_model": adv_correct,
        "adv_accuracy_on_attack_model": adv_correct / total if total else 0.0,
        "adv_accuracy_percent_on_attack_model": 100.0 * adv_correct / total if total else 0.0,
        "linf": {
            "mean": float(linf_t.mean().item()),
            "max": float(linf_t.max().item()),
            "p50": float(torch.quantile(linf_t, 0.50).item()),
            "p95": float(torch.quantile(linf_t, 0.95).item()),
        },
        "samples_csv": str(output_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

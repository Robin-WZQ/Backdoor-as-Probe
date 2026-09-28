import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .. import enforce_gpu_policy
from ..core.clip import normalize_pixels
from ..core.config import TTCConfig
from ..core.protocol import TEXT_BATCH_SIZE
from ..core.datasets import LabeledImageDirectory
from .ttc import TTCDefense

def collate_pil(batch):
    images, labels, names, _source_names = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(names)


def collate_fixed(batch):
    images, labels, names, _source_names = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(names)


@torch.no_grad()
def compute_text_features(model, processor, class_names, device, batch_size=TEXT_BATCH_SIZE, templates=None):
    templates = templates or ["a photo of a {}"]
    per_template_features = []
    for template in templates:
        prompts = [template.format(name) for name in class_names]
        chunks = []
        for start in range(0, len(prompts), batch_size):
            inputs = processor(text=prompts[start : start + batch_size], padding=True, return_tensors="pt").to(device)
            features = model.get_text_features(**inputs)
            chunks.append(F.normalize(features, dim=-1))
        per_template_features.append(torch.cat(chunks, dim=0))
    if len(per_template_features) == 1:
        return per_template_features[0]
    text_features = torch.stack(per_template_features, dim=0).mean(dim=0)
    return F.normalize(text_features, dim=-1)


def build_model(args):
    if args.defense == "none":
        model = CLIPModel.from_pretrained(args.model_id, local_files_only=args.local_files_only).to(args.device).eval()
        processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
        model.processor = processor
        return model, processor

    if args.defense == "ttc":
        config = TTCConfig(
            eps=args.ttc_eps,
            alpha=args.ttc_alpha,
            steps=args.ttc_steps,
            tau=args.ttc_tau,
            beta=args.ttc_beta,
        )
        model = TTCDefense(
            args.model_id,
            config=config,
            device=args.device,
            local_files_only=args.local_files_only,
        ).eval()
        return model, model.processor

    raise ValueError(f"Unsupported defense: {args.defense}")


def evaluate(args):
    enforce_gpu_policy(args.device)
    dataset = LabeledImageDirectory(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        dataset_name=args.dataset_name,
        classes_json=args.classes_json,
        image_size=args.image_size,
        limit=args.limit,
        return_pil=args.image_preprocess == "processor",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil if args.image_preprocess == "processor" else collate_fixed,
    )
    model, processor = build_model(args)
    text_features = compute_text_features(
        model,
        processor,
        dataset.class_names,
        args.device,
        templates=parse_templates(args.prompt_templates),
    )

    rows = []
    correct = 0
    total = 0
    use_raw_pixels = args.defense != "none"

    for images, labels, names in tqdm(loader, desc=f"Evaluating {args.defense}"):
        labels = labels.to(args.device)
        if args.image_preprocess == "processor":
            processor_kwargs = {"images": images, "return_tensors": "pt"}
            if use_raw_pixels:
                processor_kwargs["do_normalize"] = False
            pixel_values = processor(**processor_kwargs).pixel_values.to(args.device)
        else:
            pixel_values = images.to(args.device)
            if args.defense == "none":
                pixel_values = normalize_pixels(pixel_values)

        with torch.set_grad_enabled(args.defense != "none"):
            image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ text_features.T
        predictions = logits.argmax(dim=-1)

        batch_correct = predictions.eq(labels)
        correct += int(batch_correct.sum().item())
        total += int(labels.numel())

        for name, label, pred, ok in zip(names, labels.tolist(), predictions.tolist(), batch_correct.tolist()):
            rows.append(
                {
                    "image_name": name,
                    "label_idx": label,
                    "label": dataset.class_names[label],
                    "pred_idx": pred,
                    "pred": dataset.class_names[pred],
                    "correct": int(ok),
                }
            )

    accuracy = correct / total if total else 0.0
    result = {
        "image_dir": args.image_dir,
        "dataset_name": args.dataset_name,
        "labels_csv": args.labels_csv,
        "model_id": args.model_id,
        "defense": args.defense,
        "image_preprocess": args.image_preprocess,
        "prompt_templates": parse_templates(args.prompt_templates),
        "text_batch_size": TEXT_BATCH_SIZE,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "ttc": {
            "eps": args.ttc_eps,
            "alpha": args.ttc_alpha,
            "steps": args.ttc_steps,
            "tau": args.ttc_tau,
            "beta": args.ttc_beta,
        }
        if args.defense == "ttc"
        else None,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot top-1 accuracy on a labeled image directory.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--classes_json", default=None)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument(
        "--model_id",
        default=os.environ.get("BAP_BASE_MODEL_ID", "openai/clip-vit-base-patch16"),
    )
    parser.add_argument("--defense", choices=["none", "ttc"], default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--image_preprocess",
        choices=("processor", "fixed_224"),
        default="processor",
        help="Use CLIPProcessor geometry or the fixed 224x224 attack/probe geometry.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--ttc_eps", type=float, default=4 / 255)
    parser.add_argument("--ttc_alpha", type=float, default=2 / 255)
    parser.add_argument("--ttc_steps", type=int, default=3)
    parser.add_argument("--ttc_tau", type=float, default=0.3)
    parser.add_argument("--ttc_beta", type=float, default=2.0)
    parser.add_argument(
        "--prompt_templates",
        default="a photo of a {}",
        help="Prompt template string, or a JSON list of templates.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def parse_templates(value: str):
    value = value.strip()
    if value.startswith("["):
        templates = json.loads(value)
    else:
        templates = [value]
    if not templates or any("{}" not in template for template in templates):
        raise ValueError("Each prompt template must contain '{}'.")
    return templates


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    evaluate(parse_args())

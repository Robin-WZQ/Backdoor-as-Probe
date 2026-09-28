import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from transformers import CLIPModel, CLIPProcessor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

def normalize_pixels(pixel_values: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=pixel_values.device).view(1, 3, 1, 1)
    return (pixel_values - mean) / std



def default_imagenet_classes() -> List[str]:
    return list(ResNet50_Weights.DEFAULT.meta["categories"])


def build_label_mapping(labels: Sequence[str], dataset_name: str = "") -> Tuple[List[str], dict]:
    if "imagenet" in dataset_name.lower():
        all_classes = default_imagenet_classes()
        mapping = {}
        for idx, class_string in enumerate(all_classes):
            mapping[class_string] = idx
            for alias in class_string.split(","):
                mapping[alias.strip()] = idx
        return all_classes, mapping

    all_classes = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(all_classes)}
    return all_classes, mapping


class LabeledImageCSVDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        transform: transforms.Compose | None = None,
        sample_n: int | None = None,
        dataset_name: str = "",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.csv_path = self.dataset_root / "labels.csv"
        self.transform = transform or build_image_transform()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"labels.csv not found under {self.dataset_root}")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty labels.csv: {self.csv_path}")

        header = rows[0]
        data_rows = rows[1:] if len(header) >= 2 and any("label" in c.lower() for c in header) else rows
        if sample_n is not None and len(data_rows) > sample_n:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(data_rows), size=sample_n, replace=False)
            data_rows = [data_rows[i] for i in indices]

        self.image_files = [row[0] for row in data_rows]
        self.label_texts = [row[1] for row in data_rows]
        self.all_classes, self.class_to_idx = build_label_mapping(self.label_texts, dataset_name=dataset_name)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        file_name = self.image_files[index]
        label_text = self.label_texts[index]
        label_idx = self.class_to_idx.get(label_text)
        if label_idx is None:
            for k, v in self.class_to_idx.items():
                if label_text in k or k in label_text:
                    label_idx = v
                    break
        if label_idx is None:
            label_idx = 0

        image_path = self.images_dir / file_name
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label_idx, file_name


class DirectoryImageDataset(Dataset):
    def __init__(self, image_dir: str, transform: transforms.Compose | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform or build_image_transform()
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )
        if not self.image_paths:
            raise ValueError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, image_path.name


@dataclass
class LoadedCLIP:
    model: CLIPModel
    processor: CLIPProcessor
    device: torch.device


def load_clip_model(
    model_id_or_path: str,
    device: str | torch.device,
    *,
    local_files_only: bool = False,
) -> LoadedCLIP:
    device = torch.device(device)
    model = CLIPModel.from_pretrained(
        model_id_or_path, local_files_only=local_files_only
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        model_id_or_path, local_files_only=local_files_only
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return LoadedCLIP(model=model, processor=processor, device=device)


def encode_text_features(
    loaded: LoadedCLIP,
    class_names: Sequence[str],
    prompt_template: str = "a photo of a {}",
    batch_size: int = 100,
) -> torch.Tensor:
    prompts = [prompt_template.format(name) for name in class_names]
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = loaded.processor(text=batch, return_tensors="pt", padding=True).to(loaded.device)
            text_features = loaded.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
            outputs.append(text_features)
    return torch.cat(outputs, dim=0)


def extract_pooled_features(loaded: LoadedCLIP, pixel_values: torch.Tensor) -> torch.Tensor:
    pixel_values = normalize_pixels(pixel_values.to(loaded.device))
    vision_outputs = loaded.model.vision_model(pixel_values=pixel_values)
    return vision_outputs.pooler_output


def extract_vision_features(
    loaded: LoadedCLIP,
    pixel_values: torch.Tensor,
    feature_layer: int | None = None,
) -> torch.Tensor:
    pixel_values = normalize_pixels(pixel_values.to(loaded.device))
    if feature_layer is None:
        vision_outputs = loaded.model.vision_model(pixel_values=pixel_values)
        return vision_outputs.pooler_output

    vision_outputs = loaded.model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
    hidden_states = vision_outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Vision model did not return hidden states.")

    num_encoder_layers = len(hidden_states) - 1
    if feature_layer < 0 or feature_layer >= num_encoder_layers:
        raise ValueError(f"feature_layer must be in [0, {num_encoder_layers - 1}], got {feature_layer}.")

    # hidden_states[0] is the patch+position embedding output; encoder layer i lives at hidden_states[i + 1].
    return hidden_states[feature_layer + 1][:, 0, :]


def iter_batches(dataset: Dataset, batch_size: int):
    total = len(dataset)
    for start in range(0, total, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, total))]
        yield items


def stack_first(items: Iterable[tuple]) -> torch.Tensor:
    return torch.stack([x[0] for x in items], dim=0)

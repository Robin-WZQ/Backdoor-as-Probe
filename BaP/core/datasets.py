from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights

from .clip import IMAGE_EXTS, build_image_transform


@dataclass(frozen=True)
class DatasetSpec:
    category: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.category}/{self.name}"

    @property
    def slug(self) -> str:
        return f"{self.category.lower()}_{self.name.lower()}"


CSR_DATASETS = (
    DatasetSpec("General", "ImageNet"),
    DatasetSpec("General", "CIFAR10"),
    DatasetSpec("General", "CIFAR100"),
    DatasetSpec("General", "STL10"),
    DatasetSpec("General", "Caltech101"),
    DatasetSpec("General", "Caltech256"),
    DatasetSpec("FineGrained", "OxfordPets"),
    DatasetSpec("FineGrained", "Flowers102"),
    DatasetSpec("FineGrained", "Food101"),
    DatasetSpec("FineGrained", "StanfordCars"),
    DatasetSpec("Scene", "SUN397"),
    DatasetSpec("Scene", "Country211"),
    DatasetSpec("Domain", "FGVCAircraft"),
    DatasetSpec("Domain", "EuroSAT"),
    DatasetSpec("Domain", "DTD"),
    DatasetSpec("Domain", "PCAM"),
)

CSR_CATEGORIES = ("General", "FineGrained", "Scene", "Domain")


def dataset_spec(key: str) -> DatasetSpec:
    normalized = key.strip().replace("\\", "/")
    for spec in CSR_DATASETS:
        if normalized in {spec.key, spec.name, spec.slug}:
            return spec
    valid = ", ".join(spec.key for spec in CSR_DATASETS)
    raise KeyError(f"Unknown dataset {key!r}; expected one of: {valid}")


def read_label_rows(labels_csv: str | Path) -> list[dict[str, str]]:
    path = Path(labels_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"Empty labels CSV: {path}")
        name_key = "image" if "image" in fieldnames else "filename" if "filename" in fieldnames else None
        label_key = "label" if "label" in fieldnames else None
        label_idx_key = (
            "label_idx"
            if "label_idx" in fieldnames
            else "label_index"
            if "label_index" in fieldnames
            else None
        )
        if name_key is None or label_key is None:
            raise ValueError(f"labels CSV must contain image/filename and label columns: {path}")
        rows = []
        for row in reader:
            item = {"image": row[name_key], "label": row[label_key]}
            if label_idx_key is not None and row.get(label_idx_key, "") != "":
                item["label_idx"] = row[label_idx_key]
            rows.append(item)
    if not rows:
        raise ValueError(f"No label rows found: {path}")
    return rows


def imagenet_classes() -> list[str]:
    return list(ResNet50_Weights.DEFAULT.meta["categories"])


def infer_class_names(rows: Sequence[dict[str, str]], dataset_name: str) -> list[str]:
    if dataset_name.lower() == "imagenet":
        return imagenet_classes()
    return sorted({row["label"] for row in rows})


def class_mapping(class_names: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, class_name in enumerate(class_names):
        mapping[class_name] = idx
        mapping[class_name.lower()] = idx
        for alias in class_name.split(","):
            mapping[alias.strip()] = idx
            mapping[alias.strip().lower()] = idx
    return mapping


def resolve_label(label: str, mapping: dict[str, int]) -> int:
    for candidate in (label, label.lower()):
        if candidate in mapping:
            return mapping[candidate]
    raise KeyError(f"Could not map label {label!r} to the dataset class list")


def load_class_names(dataset_root: str | Path, dataset_name: str | None = None) -> list[str]:
    root = Path(dataset_root)
    classes_json = root / "classes.json"
    if classes_json.exists():
        value = json.loads(classes_json.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
            raise ValueError(f"Invalid classes.json: {classes_json}")
        return value
    rows = read_label_rows(root / "labels.csv")
    return infer_class_names(rows, dataset_name or root.name)


class LabeledImageDirectory(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        labels_csv: str | Path,
        dataset_name: str,
        classes_json: str | Path | None = None,
        image_size: int = 224,
        limit: int | None = None,
        return_pil: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_csv = Path(labels_csv)
        self.dataset_name = dataset_name
        self.return_pil = return_pil
        self.transform = build_image_transform(image_size)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if classes_json is None:
            candidate = self.labels_csv.parent / "classes.json"
            classes_json = candidate if candidate.exists() else None
        if classes_json is not None:
            self.class_names = json.loads(Path(classes_json).read_text(encoding="utf-8"))
        else:
            rows_for_classes = read_label_rows(self.labels_csv)
            self.class_names = infer_class_names(rows_for_classes, dataset_name)
        if not self.class_names or not all(isinstance(x, str) for x in self.class_names):
            raise ValueError(f"Invalid class list for {dataset_name}")
        self.class_to_idx = class_mapping(self.class_names)

        paths_by_name = {
            path.name: path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        }
        paths_by_stem: dict[str, list[Path]] = {}
        for path in paths_by_name.values():
            paths_by_stem.setdefault(path.stem, []).append(path)

        samples: list[tuple[Path, int, str]] = []
        matched_paths: set[Path] = set()
        for row in read_label_rows(self.labels_csv):
            source_name = row["image"]
            path = paths_by_name.get(source_name)
            if path is None:
                matches = paths_by_stem.get(Path(source_name).stem, [])
                if len(matches) == 1:
                    path = matches[0]
            if path is None:
                continue
            label_idx = resolve_label(row["label"], self.class_to_idx)
            # Source CSV indices are dataset-native metadata and do not share one
            # ordering convention. Classification always follows classes.json.
            samples.append((path, label_idx, source_name))
            matched_paths.add(path)

        unexpected = sorted(
            (path for path in paths_by_name.values() if path not in matched_paths),
            key=lambda path: path.name,
        )
        if unexpected:
            preview = ", ".join(path.name for path in unexpected[:8])
            suffix = "..." if len(unexpected) > 8 else ""
            raise ValueError(
                f"{len(unexpected)} images have no matching label: {preview}{suffix}"
            )
        self.samples = samples[:limit] if limit is not None else samples
        if not self.samples:
            raise ValueError(f"No labeled images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label_idx, source_name = self.samples[index]
        image = Image.open(path).convert("RGB")
        value = image if self.return_pil else self.transform(image)
        return value, label_idx, path.name, source_name

    def batch(self, start: int, batch_size: int):
        items = [self[i] for i in range(start, min(start + batch_size, len(self)))]
        images = torch.stack([item[0] for item in items], dim=0)
        labels = torch.tensor([item[1] for item in items], dtype=torch.long)
        names = [item[2] for item in items]
        source_names = [item[3] for item in items]
        return images, labels, names, source_names

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


REQUIRED_COLUMNS = ("content_path", "style_path", "reference_path", "legacy_path")


class LegacyManifestDataset(Dataset):
    """CSV manifest dataset shared by all FFG adapters."""

    def __init__(self, manifest_path: str, image_size: int):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.rows = self._load_rows()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _load_rows(self) -> List[Dict[str, str]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
            )

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Manifest is missing columns: {missing}")
            return [row for row in reader if any((value or "").strip() for value in row.values())]

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.manifest_path.parent / path).resolve()

    def _load_image(self, value: str) -> torch.Tensor:
        path = self._resolve(value)
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[index]
        target_path = (row.get("target_path") or "").strip() or row["legacy_path"]

        sample: Dict[str, torch.Tensor | str] = {
            "char": row.get("char", ""),
            "content_image": self._load_image(row["content_path"]),
            "style_image": self._load_image(row["style_path"]),
            "reference_image": self._load_image(row["reference_path"]),
            "legacy_image": self._load_image(row["legacy_path"]),
            "target_image": self._load_image(target_path),
            "content_path": str(self._resolve(row["content_path"])),
            "style_path": str(self._resolve(row["style_path"])),
            "reference_path": str(self._resolve(row["reference_path"])),
            "legacy_path": str(self._resolve(row["legacy_path"])),
            "target_path": str(self._resolve(target_path)),
        }
        return sample

    def __len__(self) -> int:
        return len(self.rows)

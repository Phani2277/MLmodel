import csv
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class CarStateDataset(Dataset):
    """Dataset reading images and binary labels from a CSV file."""

    def __init__(self, annotations_file: str, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.items: list[Tuple[Path, torch.Tensor]] = []

        with open(annotations_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = self.root_dir / row["filepath"]
                labels = torch.tensor([int(row["dirty"]), int(row["damaged"])], dtype=torch.float32)
                self.items.append((path, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, labels = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, labels

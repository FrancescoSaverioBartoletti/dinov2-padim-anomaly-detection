from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import torch
from PIL import Image


class CropSides:
    def __init__(self, left=0, right=0, top=0, bottom=0):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((self.left, self.top, w - self.right, h - self.bottom))


class ToothbrushImageDataset(Dataset):

    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(Path(targ_dir).glob("*/*.png"))

        self.classes = sorted({path.parent.name for path in self.paths})
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        self.labels = [self.class_to_idx[path.parent.name] for path in self.paths]
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index]).convert("RGB")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        label = self.labels[index]

        if self.transform:
            return self.transform(img), label
        return img, label

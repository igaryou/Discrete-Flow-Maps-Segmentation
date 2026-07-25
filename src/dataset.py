from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF


ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)
for cityscapes_id, train_id in {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}.items():
    ID_TO_20CLASS[cityscapes_id] = train_id


class Cityscapes20ClassDataset(Dataset):
    """Cityscapes with 19 semantic classes plus void at class index 19."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: list[int] | tuple[int, int] | None = None,
        crop_size: list[int] | tuple[int, int] | None = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        imagenet_normalize: bool = False,
    ) -> None:
        self.image_size = tuple(image_size) if image_size is not None else None
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter_enabled = color_jitter
        self.imagenet_normalize = imagenet_normalize
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.dataset = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        image = TF.pil_to_tensor(image).float() / 255.0
        mask = torch.from_numpy(ID_TO_20CLASS[np.asarray(target, dtype=np.uint8)]).long()

        if self.augment and torch.rand(()) < self.hflip_prob:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))
        if self.image_size is not None:
            image = TF.resize(
                image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resize(
                mask[None], self.image_size, interpolation=TF.InterpolationMode.NEAREST
            )[0].long()
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            _, height, width = image.shape
            if crop_h > height or crop_w > width:
                raise ValueError("dataset.crop_size exceeds the resized image")
            top = torch.randint(0, height - crop_h + 1, ()).item()
            left = torch.randint(0, width - crop_w + 1, ()).item()
            image = TF.crop(image, top, left, crop_h, crop_w)
            mask = TF.crop(mask, top, left, crop_h, crop_w)
        if self.augment and self.color_jitter_enabled:
            image = self.jitter(image).clamp(0.0, 1.0)
        if self.imagenet_normalize:
            mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
            std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            image = (image - mean) / std

        one_hot = F.one_hot(mask, num_classes=20).permute(2, 0, 1).float()
        return image, one_hot, mask


def build_dataset(config: dict, split: str, augment: bool | None = None):
    aug = config["augmentation"]
    flip = aug["horizontal_flip"]
    jitter = aug["color_jitter"]
    enabled = aug["enabled"] if augment is None else augment
    return Cityscapes20ClassDataset(
        root=config["dataset"]["root"],
        split=split,
        image_size=config["dataset"]["image_size"],
        crop_size=config["dataset"]["crop_size"] if split == "train" else None,
        augment=enabled and split == "train",
        hflip_prob=flip["probability"] if flip["enabled"] else 0.0,
        color_jitter=enabled and jitter["enabled"],
        brightness=jitter["brightness"],
        contrast=jitter["contrast"],
        saturation=jitter["saturation"],
        hue=jitter["hue"],
        imagenet_normalize=aug["imagenet_normalize"],
    )


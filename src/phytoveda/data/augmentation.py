"""Data augmentation pipeline leveraging Python 3.14 free-threading.

Augmentations applied on-the-fly during training:
    - Multi-angle rotations (90, 180, 270 + random)
    - Horizontal flipping
    - Zooming / random cropping
    - Brightness and contrast adjustment
    - Gaussian noise injection
    - Color jittering (H/S/V)
    - Affine transformations
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Training augmentation pipeline.

    Designed to be CPU-intensive — relies on Python 3.14 free-threading
    to saturate all cores via DataLoader num_workers.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-15, 15), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Validation/test transforms — resize and normalize only."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

"""Image preprocessing: resize to 512x512, normalize, handle corrupt images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_SIZE = 512


def preprocess_image(image_path: str | Path) -> Image.Image | None:
    """Load and preprocess a single image to 512x512 RGB.

    Returns None for corrupt or unreadable images.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return img
    except (OSError, Image.DecompressionBombError):
        return None


def validate_image(image_path: str | Path) -> bool:
    """Check if an image file is valid and readable."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

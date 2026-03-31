"""Dataset download and validation utilities.

Sources:
    - Herbify: GitHub (Phantom-fs/Herbify-Dataset)
    - Assam (MED117): Mendeley Data (dtvbwrhznz/4)
    - AI-MedLeafX: Mendeley Data (zz7r5y4dc6/1)
    - CIMPD: Kaggle (cimpd dataset)
    - SIMP: Mendeley Data (9d89vjcghv/2)
    - EarlyNSD: Kaggle (raiaone/early-nutrient-stress-detection-of-plants)
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetSource:
    """Download metadata for a single dataset."""

    name: str
    platform: str  # "kaggle", "mendeley", "github"
    identifier: str  # Kaggle slug, Mendeley DOI, or GitHub repo
    expected_images: int
    description: str


DATASET_SOURCES: dict[str, DatasetSource] = {
    "herbify": DatasetSource(
        name="Herbify",
        platform="github",
        identifier="Phantom-fs/Herbify-Dataset",
        expected_images=6_104,
        description="91 medicinal herb species, healthy baselines with metadata",
    ),
    "assam": DatasetSource(
        name="Assam Medicinal Leaf Set (MED117)",
        platform="mendeley",
        identifier="dtvbwrhznz/4",
        expected_images=7_341,
        description="10 evaluation classes from NE India, regional morphological variance",
    ),
    "medleafx": DatasetSource(
        name="AI-MedLeafX",
        platform="mendeley",
        identifier="zz7r5y4dc6/1",
        expected_images=10_858,
        description=(
            "4 species with disease labels: Bacterial Spot, "
            "Shot Hole, Powdery Mildew, Yellow Leaf"
        ),
    ),
    "cimpd": DatasetSource(
        name="CIMPD",
        platform="kaggle",
        identifier="cimpd-central-india-medicinal-plant",
        expected_images=9_130,
        description="23 species, Healthy/Unhealthy labels, unconstrained smartphone capture",
    ),
    "simp": DatasetSource(
        name="SIMP",
        platform="mendeley",
        identifier="9d89vjcghv/2",
        expected_images=2_503,
        description="20 species, herbs/shrubs/creepers/climbers/trees from South India",
    ),
    "earlynsd": DatasetSource(
        name="EarlyNSD",
        platform="kaggle",
        identifier="raiaone/early-nutrient-stress-detection-of-plants",
        expected_images=2_700,
        description="9 classes: 3 cucurbits x (Healthy, N-deficiency, K-deficiency)",
    ),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def count_images(directory: Path) -> int:
    """Count image files recursively in a directory."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(data_root: Path, dataset_key: str) -> tuple[bool, str]:
    """Validate that a downloaded dataset has the expected number of images.

    Returns (is_valid, message).
    """
    source = DATASET_SOURCES[dataset_key]
    dataset_dir = data_root / dataset_key

    if not dataset_dir.exists():
        return False, f"{source.name}: directory not found at {dataset_dir}"

    image_count = count_images(dataset_dir)
    # Allow 5% tolerance for minor dataset version differences
    min_expected = int(source.expected_images * 0.90)

    if image_count < min_expected:
        return False, (
            f"{source.name}: found {image_count} images, "
            f"expected ~{source.expected_images} (min {min_expected})"
        )

    return True, f"{source.name}: {image_count} images validated"


def validate_all(data_root: Path) -> dict[str, tuple[bool, str]]:
    """Validate all datasets under data_root."""
    results = {}
    for key in DATASET_SOURCES:
        results[key] = validate_dataset(data_root, key)
    return results


def download_kaggle(identifier: str, output_dir: Path) -> None:
    """Download a dataset from Kaggle using the kaggle CLI.

    Requires: pip install kaggle && kaggle API key configured.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", identifier,
            "-p", str(output_dir),
            "--unzip",
        ],
        check=True,
    )


def download_github(repo: str, output_dir: Path) -> None:
    """Clone a GitHub repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(output_dir)],
        check=True,
    )


def download_mendeley(identifier: str, output_dir: Path) -> None:
    """Download from Mendeley Data.

    Mendeley datasets don't have a standard CLI — prints instructions
    for manual download, or uses the Mendeley Data API if available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mendeley_url = f"https://data.mendeley.com/datasets/{identifier}"
    print(
        f"Mendeley dataset: {identifier}\n"
        f"Please download manually from: {mendeley_url}\n"
        f"Extract contents to: {output_dir}\n"
    )


def download_dataset(data_root: Path, dataset_key: str) -> None:
    """Download a single dataset to data_root/<dataset_key>/."""
    source = DATASET_SOURCES[dataset_key]
    output_dir = data_root / dataset_key

    if output_dir.exists() and count_images(output_dir) > 0:
        print(f"  {source.name}: already exists with {count_images(output_dir)} images, skipping")
        return

    print(f"  Downloading {source.name} from {source.platform}...")

    if source.platform == "kaggle":
        download_kaggle(source.identifier, output_dir)
    elif source.platform == "github":
        download_github(source.identifier, output_dir)
    elif source.platform == "mendeley":
        download_mendeley(source.identifier, output_dir)

    is_valid, msg = validate_dataset(data_root, dataset_key)
    print(f"  {msg}")


def download_all(data_root: str | Path) -> None:
    """Download all 6 datasets."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("PhytoVeda Dataset Download")
    print("=" * 60)

    for key, source in DATASET_SOURCES.items():
        print(f"\n[{key}] {source.name} ({source.expected_images} images)")
        download_dataset(data_root, key)

    print("\n" + "=" * 60)
    print("Validation Summary:")
    results = validate_all(data_root)
    for _key, (is_valid, msg) in results.items():
        status = "OK" if is_valid else "MISSING"
        print(f"  [{status}] {msg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PhytoVeda datasets")
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root directory for datasets (default: data)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASET_SOURCES.keys()),
        help="Download a specific dataset (default: all)",
    )
    args = parser.parse_args()

    if args.dataset:
        download_dataset(Path(args.data_root), args.dataset)
    else:
        download_all(args.data_root)

"""Dataset download and validation utilities.

Sources:
    - Herbify: GitHub (Phantom-fs/Herbify-Dataset)
    - Assam (MED117): Mendeley Data (dtvbwrhznz/4)
    - AI-MedLeafX: Mendeley Data (zz7r5y4dc6/1)
    - CIMPD: Kaggle (satyamtomar08/indian-medicinal-plant-dataset)
    - SIMPD V1 (South Indian Medicinal Plants): Mendeley
      https://data.mendeley.com/datasets/9d89vjcghv/2
    - EarlyNSD: Kaggle (raiaone/early-nutrient-stress-detection-of-plants)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Shared HTTP opener with browser-like headers (Mendeley/Cloudfront reject
# bare urllib requests without a User-Agent).
_URL_OPENER = urllib.request.build_opener()
_URL_OPENER.addheaders = [
    ("User-Agent", "Mozilla/5.0 (PhytoVeda Dataset Downloader)"),
    ("Accept", "*/*"),
]


@dataclass
class DatasetSource:
    """Download metadata for a single dataset."""

    name: str
    platform: str  # "kaggle", "mendeley", "github"
    identifier: str  # Kaggle slug, Mendeley DOI, or GitHub repo
    expected_images: int
    description: str
    zip_keywords: tuple[str, ...]  # Case-insensitive substrings to match local ZIPs


DATASET_SOURCES: dict[str, DatasetSource] = {
    "herbify": DatasetSource(
        name="Herbify",
        platform="github",
        identifier="Phantom-fs/Herbify-Dataset",
        expected_images=6_104,
        description="91 medicinal herb species, healthy baselines with metadata",
        zip_keywords=("herbify",),
    ),
    "assam": DatasetSource(
        name="Assam Medicinal Leaf Set (MED117)",
        platform="mendeley",
        identifier="dtvbwrhznz/4",
        expected_images=7_341,
        description="10 evaluation classes from NE India, regional morphological variance",
        zip_keywords=("med117", "medicinal plant leaf"),
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
        zip_keywords=("medleafx", "ai-medleafx"),
    ),
    "cimpd": DatasetSource(
        name="CIMPD (Central India Medicinal Plant Dataset)",
        platform="kaggle",
        identifier="satyamtomar08/indian-medicinal-plant-dataset",
        expected_images=9_130,
        description="23 species from Gwalior region, leaf images for species classification",
        zip_keywords=("cimpd",),
    ),
    "simp": DatasetSource(
        name="SIMPD V1 (South Indian Medicinal Plants)",
        platform="mendeley",
        identifier="9d89vjcghv/2",
        expected_images=2_503,
        description=(
            "SIMPD V1: 2,503 wild-scene images, 20 classes (herbs, shrubs, creepers, "
            "climbers, trees); source https://data.mendeley.com/datasets/9d89vjcghv/2"
        ),
        zip_keywords=("simpd", "south indian medicinal"),
    ),
    "earlynsd": DatasetSource(
        name="EarlyNSD",
        platform="kaggle",
        identifier="raiaone/early-nutrient-stress-detection-of-plants",
        expected_images=2_700,
        description="9 classes: 3 cucurbits x (Healthy, N-deficiency, K-deficiency)",
        zip_keywords=("early_nsd", "earlynsd", "nutrient stress"),
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
    """Download a dataset from Kaggle via ``kagglehub``.

    ``kagglehub`` will prompt for login interactively if no credentials are
    found (Colab secrets, ``~/.kaggle/kaggle.json``, or env vars).

    Requires: ``pip install kagglehub``

    Raises:
        RuntimeError: If kagglehub is not installed or the download fails.
    """
    try:
        import kagglehub  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # kagglehub.dataset_download returns the path to its local cache.
    # It handles auth prompts, progress bars, and caching automatically.
    cached_path = kagglehub.dataset_download(identifier)
    cached = Path(cached_path)

    # Copy from kagglehub cache into our expected output directory
    if cached.resolve() != output_dir.resolve():
        shutil.copytree(cached, output_dir, dirs_exist_ok=True)


def download_github(repo: str, output_dir: Path) -> None:
    """Clone a GitHub repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(output_dir)],
        check=True,
    )


def _fetch_url(url: str, dest: Path) -> None:
    """Download *url* to *dest* using the shared opener (sends User-Agent)."""
    with _URL_OPENER.open(url, timeout=300) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _mendeley_via_api_files(dataset_id: str, version: str, output_dir: Path) -> bool:
    """List individual files via the Mendeley Data REST API and download each."""
    api_url = (
        f"https://data.mendeley.com/api/datasets-v2/datasets/{dataset_id}"
        f"?version={version}"
    )
    try:
        with _URL_OPENER.open(api_url, timeout=60) as resp:
            meta = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False

    files = meta.get("files", [])
    if not files:
        return False

    for fmeta in files:
        fname = fmeta.get("filename") or fmeta.get("name", "unknown")
        dl_url = fmeta.get("download_url") or fmeta.get("content_details", {}).get("download_url")
        if not dl_url:
            continue
        dest = output_dir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"    Downloading {fname}...")
        try:
            _fetch_url(dl_url, dest)
        except (urllib.error.URLError, OSError) as exc:
            print(f"    Warning: failed to download {fname}: {exc}")
            continue
        # Auto-extract ZIP files (handles nested ZIPs too)
        if dest.suffix.lower() == ".zip" and zipfile.is_zipfile(dest):
            _extract_recursive(dest, output_dir)
            dest.unlink(missing_ok=True)

    return True


def _mendeley_via_zip(dataset_id: str, version: str, output_dir: Path) -> bool:
    """Try downloading the whole dataset as a single ZIP."""
    zip_urls = [
        f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files/dataset-{version}.zip",
        f"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/{dataset_id}-{version}.zip",
        f"https://data.mendeley.com/api/datasets-v2/datasets/{dataset_id}/zip?version={version}",
    ]
    for url in zip_urls:
        tmp_path = Path(tempfile.mktemp(suffix=".zip"))
        try:
            _fetch_url(url, tmp_path)
            if not zipfile.is_zipfile(tmp_path):
                tmp_path.unlink(missing_ok=True)
                continue
            _extract_recursive(tmp_path, output_dir)
            tmp_path.unlink(missing_ok=True)
            return True
        except (urllib.error.URLError, OSError, zipfile.BadZipFile):
            tmp_path.unlink(missing_ok=True)
    return False


def download_mendeley(identifier: str, output_dir: Path) -> None:
    """Download from Mendeley Data.

    Strategy order:
        1. REST API — list files and download each individually (most reliable).
        2. Bulk ZIP — try known ZIP endpoints with User-Agent headers.
        3. Manual fallback — print instructions for the user.

    Args:
        identifier: Mendeley dataset ID in ``{id}/{version}`` format
                    (e.g. ``dtvbwrhznz/4``).
        output_dir: Directory to extract the dataset into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id, _, version = identifier.partition("/")
    version = version or "1"

    # Strategy 1: REST API per-file download
    print("  Trying Mendeley REST API (per-file)...")
    if _mendeley_via_api_files(dataset_id, version, output_dir):
        print("  Downloaded via Mendeley REST API")
        return

    # Strategy 2: Bulk ZIP download
    print("  Trying bulk ZIP download...")
    if _mendeley_via_zip(dataset_id, version, output_dir):
        print("  Downloaded via bulk ZIP")
        return

    # Strategy 3: Manual fallback
    mendeley_url = f"https://data.mendeley.com/datasets/{identifier}"
    print(
        f"  Automatic download failed. Please download manually:\n"
        f"    {mendeley_url}\n"
        f"  Extract contents to: {output_dir}\n"
    )


def _extract_recursive(zip_path: Path, output_dir: Path) -> None:
    """Extract a ZIP into *output_dir*, recursively extracting any inner ZIPs.

    Handles structures like AI-MedLeafX where the outer ZIP contains
    ``Original.zip`` and ``Augmented.zip`` inside it.

    After extraction:
    - All inner ZIPs are extracted in-place and then deleted.
    - If the result is a single top-level folder, its contents are
      flattened up one level so dataset loaders find species folders
      directly under *output_dir*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Recursively extract any inner ZIPs that appeared
    inner_zips = list(output_dir.rglob("*.zip"))
    for inner in inner_zips:
        if not zipfile.is_zipfile(inner):
            continue
        # Extract next to the inner ZIP, then remove it
        inner_dest = inner.parent / inner.stem
        print(f"    Extracting inner ZIP: {inner.name} ...")
        _extract_recursive(inner, inner_dest)
        inner.unlink()

    # Flatten: if extraction produced a single top-level folder, hoist
    # its contents up so loaders see species dirs directly.
    _flatten_single_child(output_dir)


def _flatten_single_child(directory: Path) -> None:
    """If *directory* contains exactly one non-hidden child dir, hoist its
    contents up one level.  Applied recursively until the pattern breaks."""
    children = [p for p in directory.iterdir() if not p.name.startswith(".")]
    if len(children) != 1 or not children[0].is_dir():
        return

    nested = children[0]
    for item in list(nested.iterdir()):
        dest = directory / item.name
        if not dest.exists():
            item.rename(dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            shutil.rmtree(item)
        else:
            item.replace(dest)
    if nested.exists():
        shutil.rmtree(nested, ignore_errors=True)

    # Check again — the hoisted content might itself be a single folder
    _flatten_single_child(directory)


def _try_extract_local_zip(
    data_root: Path,
    dataset_key: str,
    zip_dirs: list[Path] | None = None,
) -> bool:
    """Scan directories for a ZIP file matching *dataset_key* and extract it.

    Searches *data_root* first, then each path in *zip_dirs* (e.g. a Google
    Drive folder with manually downloaded ZIPs).

    Matches are found by checking each ``.zip`` filename (case-insensitive)
    against the ``zip_keywords`` defined in the dataset's ``DatasetSource``.

    Returns ``True`` if a matching ZIP was found and extracted.
    """
    source = DATASET_SOURCES[dataset_key]
    output_dir = data_root / dataset_key

    search_dirs = [data_root] + (zip_dirs or [])

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Search recursively so ZIPs in subdirectories are found too
        # (e.g. PhytoVedaData/subdir/dataset.zip)
        for zpath in search_dir.rglob("*.zip"):
            if not zpath.is_file():
                continue
            fname_lower = zpath.stem.lower()
            if any(kw in fname_lower for kw in source.zip_keywords):
                print(f"  Found local ZIP: {zpath}")
                print(f"  Extracting to {output_dir} ...")
                _extract_recursive(zpath, output_dir)
                print(f"  Extracted {zpath.name}")
                return True

    return False


def download_dataset(
    data_root: Path,
    dataset_key: str,
    zip_dirs: list[Path] | None = None,
) -> bool:
    """Download a single dataset to ``data_root/<dataset_key>/``.

    Before attempting a remote download, checks for a matching ``.zip`` file
    in *data_root* and any extra *zip_dirs* (e.g. a Google Drive folder).

    Returns:
        ``True`` if the dataset was downloaded (or already present) and valid.
    """
    source = DATASET_SOURCES[dataset_key]
    output_dir = data_root / dataset_key

    if output_dir.exists() and count_images(output_dir) > 0:
        print(f"  {source.name}: already exists with {count_images(output_dir)} images, skipping")
        return True

    # Check for a local ZIP before hitting the network
    if _try_extract_local_zip(data_root, dataset_key, zip_dirs=zip_dirs):
        is_valid, msg = validate_dataset(data_root, dataset_key)
        print(f"  {msg}")
        return is_valid

    print(f"  Downloading {source.name} from {source.platform}...")

    try:
        if source.platform == "kaggle":
            download_kaggle(source.identifier, output_dir)
        elif source.platform == "github":
            download_github(source.identifier, output_dir)
        elif source.platform == "mendeley":
            download_mendeley(source.identifier, output_dir)
    except (RuntimeError, OSError) as exc:
        print(f"  ERROR downloading {source.name}: {exc}")
        return False

    is_valid, msg = validate_dataset(data_root, dataset_key)
    print(f"  {msg}")
    return is_valid


def download_all(
    data_root: str | Path,
    zip_dirs: list[str | Path] | None = None,
) -> None:
    """Download all 6 datasets.

    Args:
        data_root: Directory where extracted datasets are stored.
        zip_dirs:  Extra directories to scan for pre-downloaded ZIP files
                   (e.g. ``["/content/drive/MyDrive/PhytoVedaData"]``).
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    resolved_zip_dirs = [Path(d) for d in zip_dirs] if zip_dirs else None

    print("PhytoVeda Dataset Download")
    print("=" * 60)

    for key, source in DATASET_SOURCES.items():
        print(f"\n[{key}] {source.name} ({source.expected_images} images)")
        download_dataset(data_root, key, zip_dirs=resolved_zip_dirs)

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
    parser.add_argument(
        "--zip-dir", type=str, action="append", default=None,
        help="Extra directory to scan for pre-downloaded ZIPs (repeatable)",
    )
    args = parser.parse_args()

    zip_dirs = [Path(d) for d in args.zip_dir] if args.zip_dir else None

    if args.dataset:
        download_dataset(Path(args.data_root), args.dataset, zip_dirs=zip_dirs)
    else:
        download_all(args.data_root, zip_dirs=zip_dirs)

"""Drive-backed dataset setup for Google Colab.

Orchestrates extracting pre-downloaded dataset ZIPs from Google Drive into
the Colab SSD for fast I/O during training.  Handles nested ZIPs (e.g.
AI-MedLeafX ships ``Original.zip`` + ``Augmented.zip`` inside its outer
ZIP), single-folder flattening, and keyword-based filename matching so
datasets don't need manual renaming.

Usage (Colab cell)::

    from google.colab import drive
    drive.mount("/content/drive")

    from phytoveda.colab.dataset_setup import setup_datasets
    results = setup_datasets()
    # Done — datasets extracted to /content/datasets/<key>/

Or with custom paths::

    results = setup_datasets(
        data_root="/content/datasets",
        zip_dirs=[
            "/content/drive/MyDrive/PhytoVedaData",
            "/content/drive/MyDrive/ExtraZips",
        ],
    )

Expected Drive layout
---------------------
Place your dataset ZIPs in ``/content/drive/MyDrive/PhytoVedaData/``
(or any folder you specify via *zip_dirs*).  No renaming needed — files
are matched by keyword::

    /content/drive/MyDrive/PhytoVedaData/
    ├── MED117_Medicinal Plant Leaf Dataset & Name Table.zip   → assam/
    ├── AI-MedLeafX A Large-Scale Computer Vision ...zip       → medleafx/
    ├── cimpd.zip                                               → cimpd/
    ├── SIMPD V1 South Indian Medicinal Plants ...zip           → simp/
    └── early_nsd_1.zip                                         → earlynsd/

Herbify is cloned from GitHub automatically (no ZIP needed).

ZIP matching keywords
---------------------
Each dataset is identified by case-insensitive substring matching on the
ZIP filename (stem only, without ``.zip``):

========== ========================== ==================================
Dataset    Keywords                   Example filename
========== ========================== ==================================
herbify    ``herbify``                *(auto-cloned from GitHub)*
assam      ``med117``,                ``MED117_Medicinal Plant Leaf
           ``medicinal plant leaf``   Dataset & Name Table.zip``
medleafx   ``medleafx``,              ``AI-MedLeafX A Large-Scale
           ``ai-medleafx``            Computer Vision Dataset...zip``
cimpd      ``cimpd``                  ``cimpd.zip``
simp       ``simpd``,                 ``SIMPD V1 South Indian
           ``south indian medicinal`` Medicinal Plants...zip``
earlynsd   ``early_nsd``,             ``early_nsd_1.zip``
           ``earlynsd``,
           ``nutrient stress``
========== ========================== ==================================

Nested ZIP handling
-------------------
Some datasets ship as a ZIP-of-ZIPs.  For example, AI-MedLeafX contains
``Original.zip`` and ``Augmented.zip`` inside the outer archive.  The
extractor handles this automatically:

1. Outer ZIP is extracted to ``data_root/<key>/``.
2. Any ``.zip`` files found inside are recursively extracted in-place.
3. Inner ZIP files are deleted after extraction.
4. If the result is a single top-level folder, its contents are hoisted
   up so dataset loaders find species folders directly.

Output structure
----------------
After extraction, each dataset lives at ``data_root/<key>/`` with
species folders directly underneath::

    /content/datasets/
    ├── herbify/          ← git-cloned
    │   ├── Azadirachta indica/
    │   ├── Ocimum tenuiflorum/
    │   └── ...
    ├── assam/            ← from MED117...zip
    │   ├── species_1/
    │   └── ...
    ├── medleafx/         ← from AI-MedLeafX...zip (inner ZIPs extracted)
    │   ├── Neem/
    │   │   ├── Healthy/
    │   │   └── Bacterial Spot/
    │   └── ...
    ├── cimpd/            ← from cimpd.zip
    │   ├── species_1/
    │   └── ...
    ├── simp/             ← from SIMPD V1...zip
    │   ├── Betel/
    │   └── ...
    └── earlynsd/         ← from early_nsd_1.zip
        ├── ash_gourd/
        │   ├── Healthy/
        │   └── N_deficiency/
        └── ...
"""

from __future__ import annotations

from pathlib import Path

from phytoveda.data.download import (
    DATASET_SOURCES,
    download_all,
    validate_all,
)

# Default Google Drive directory for pre-downloaded dataset ZIPs.
DEFAULT_ZIP_DIR = "/content/drive/MyDrive/PhytoVedaData"

# Default Colab SSD directory where datasets are extracted for fast I/O.
DEFAULT_DATA_ROOT = "/content/datasets"


def setup_datasets(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    zip_dirs: list[str | Path] | None = None,
) -> dict[str, tuple[bool, str]]:
    """Extract datasets from Drive ZIPs and/or download missing ones.

    This is the single entry point for Colab notebooks.  It:

    1. Scans *zip_dirs* (defaults to Drive ``PhytoVedaData/``) for ZIPs.
    2. Matches each ZIP to a dataset by filename keywords.
    3. Extracts to ``data_root/<key>/``, handling nested ZIPs and
       single-folder flattening automatically.
    4. For any datasets not found as local ZIPs, falls back to remote
       download (GitHub for Herbify, Mendeley API, kagglehub).
    5. Validates all datasets and prints a summary.

    Args:
        data_root: Directory on Colab SSD for extracted datasets.
        zip_dirs:  Directories to scan for pre-downloaded ZIPs.
                   Defaults to ``["/content/drive/MyDrive/PhytoVedaData"]``.

    Returns:
        Dict mapping dataset keys to ``(is_valid, message)`` tuples.

    Example::

        from phytoveda.colab.dataset_setup import setup_datasets

        results = setup_datasets()
        # Or with custom zip location:
        results = setup_datasets(zip_dirs=["/content/drive/MyDrive/MyData"])
    """
    if zip_dirs is None:
        zip_dirs = [DEFAULT_ZIP_DIR]

    data_root = Path(data_root)

    # Print what we're scanning
    print("PhytoVeda Dataset Setup")
    print("=" * 60)
    print(f"  Data root (SSD):  {data_root}")
    for zd in zip_dirs:
        zd_path = Path(zd)
        if zd_path.exists():
            zips = [f.name for f in zd_path.iterdir()
                    if f.suffix.lower() == ".zip"]
            print(f"  ZIP source:       {zd} ({len(zips)} ZIPs found)")
            for z in sorted(zips):
                print(f"    - {z}")
        else:
            print(f"  ZIP source:       {zd} (not found)")
    print("=" * 60)

    # Run the download pipeline (local ZIP extraction → remote fallback)
    download_all(data_root, zip_dirs=zip_dirs)

    # Return validation results
    return validate_all(data_root)


def print_dataset_status(data_root: str | Path = DEFAULT_DATA_ROOT) -> None:
    """Print a quick status table of all datasets.

    Example::

        from phytoveda.colab.dataset_setup import print_dataset_status
        print_dataset_status()
    """
    data_root = Path(data_root)
    results = validate_all(data_root)
    from phytoveda.data.download import count_images

    print(f"\n{'Dataset':<12} {'Status':<10} {'Images':>8}  {'Expected':>8}  Name")
    print("-" * 70)
    for key, (is_valid, _msg) in results.items():
        source = DATASET_SOURCES[key]
        ds_dir = data_root / key
        actual = count_images(ds_dir) if ds_dir.exists() else 0
        status = "OK" if is_valid else "MISSING"
        print(f"{key:<12} {status:<10} {actual:>8}  {source.expected_images:>8}"
              f"  {source.name}")

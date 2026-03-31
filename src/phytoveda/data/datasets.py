"""Federated botanical dataset combining all 6 data sources.

Each dataset has a specific directory layout. The loaders below handle
each format and produce unified (image_path, species_id, pathology_id) samples.

Datasets:
    - Herbify: 6,104 images, 91 species — folder-per-species
    - Assam (MED117): 7,341 images, 10 classes — folder-per-species
    - AI-MedLeafX: 10,858 orig, 4 species — species/condition folders
    - CIMPD: 9,130 images, 23 species — species/{Healthy,Unhealthy} folders
    - SIMP: 2,503 images, 20 species — folder-per-species
    - EarlyNSD: 2,700 images, 9 classes — crop_condition folders
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from phytoveda.data.augmentation import get_train_transforms, get_val_transforms
from phytoveda.data.taxonomy import (
    PATHOLOGY_CLASSES,
    SpeciesTaxonomy,
    map_pathology_label,
)

if TYPE_CHECKING:
    import albumentations as A

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─── Per-Dataset Loaders ─────────────────────────────────────────────────────
# Each returns list of (image_path, species_name, raw_pathology_label_or_None)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_herbify(root: Path) -> list[tuple[Path, str, str | None]]:
    """Herbify: data/herbify/<species_name>/*.jpg — no pathology labels."""
    samples = []
    if not root.exists():
        return samples
    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        species_name = species_dir.name
        for img in species_dir.rglob("*"):
            if _is_image(img):
                samples.append((img, species_name, None))
    return samples


def load_assam(root: Path) -> list[tuple[Path, str, str | None]]:
    """Assam (MED117): data/assam/<species_name>/*.jpg — no pathology labels.

    The MED117 raw images are in folder-per-species layout.
    We use the 10 evaluation classes subset (7,341 images).
    """
    samples = []
    if not root.exists():
        return samples
    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        species_name = species_dir.name
        for img in species_dir.rglob("*"):
            if _is_image(img):
                samples.append((img, species_name, None))
    return samples


def load_medleafx(root: Path) -> list[tuple[Path, str, str | None]]:
    """AI-MedLeafX: data/medleafx/<species>/<condition>/*.jpg

    Species: Cinnamomum camphora, Terminalia chebula, Moringa oleifera, Azadirachta indica
    Conditions: Healthy, Bacterial Spot, Shot Hole, Powdery Mildew, Yellow Leaf Disease

    The dataset may also have flat species_condition folder names.
    Handles both hierarchical and flat layouts.
    """
    samples = []
    if not root.exists():
        return samples

    for top_dir in sorted(root.iterdir()):
        if not top_dir.is_dir() or top_dir.name.startswith("."):
            continue

        subdirs = [d for d in top_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Hierarchical: species/condition/images
            species_name = top_dir.name
            for condition_dir in sorted(subdirs):
                condition = condition_dir.name
                for img in condition_dir.rglob("*"):
                    if _is_image(img):
                        samples.append((img, species_name, condition))
        else:
            # Flat: species_condition/images or just species/images
            name = top_dir.name
            # Try to split species_condition
            for sep in ["___", "__", "_-_"]:
                if sep in name:
                    parts = name.split(sep, 1)
                    species_name, condition = parts[0].strip(), parts[1].strip()
                    break
            else:
                # Could be species-only or condition embedded in name
                species_name = name
                condition = None

            for img in top_dir.rglob("*"):
                if _is_image(img):
                    samples.append((img, species_name, condition))

    return samples


def load_cimpd(root: Path) -> list[tuple[Path, str, str | None]]:
    """CIMPD: data/cimpd/<species>/{Healthy,Unhealthy}/*.jpg

    Or flat: data/cimpd/<species>/*.jpg with train/test/val splits.
    Handles both layouts.
    """
    samples = []
    if not root.exists():
        return samples

    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue

        species_name = species_dir.name
        subdirs = [d for d in species_dir.iterdir() if d.is_dir()]
        health_dirs = {d.name.lower(): d for d in subdirs}

        if "healthy" in health_dirs or "unhealthy" in health_dirs:
            # Has health subfolders
            for subdir in subdirs:
                condition = subdir.name  # "Healthy" or "Unhealthy"
                for img in subdir.rglob("*"):
                    if _is_image(img):
                        samples.append((img, species_name, condition))
        elif any(d.name.lower() in ("train", "test", "val", "validation") for d in subdirs):
            # Has split subfolders — recurse into them
            for split_dir in subdirs:
                for item in sorted(split_dir.iterdir()):
                    if item.is_dir():
                        # split/species or split/condition
                        for img in item.rglob("*"):
                            if _is_image(img):
                                samples.append((img, species_name, item.name))
                    elif _is_image(item):
                        samples.append((item, species_name, None))
        else:
            # Flat images under species folder
            for img in species_dir.rglob("*"):
                if _is_image(img):
                    samples.append((img, species_name, None))

    return samples


def load_simp(root: Path) -> list[tuple[Path, str, str | None]]:
    """SIMP: data/simp/<species_name>/*.jpg — no pathology labels."""
    samples = []
    if not root.exists():
        return samples
    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        species_name = species_dir.name
        for img in species_dir.rglob("*"):
            if _is_image(img):
                samples.append((img, species_name, None))
    return samples


def load_earlynsd(root: Path) -> list[tuple[Path, str, str | None]]:
    """EarlyNSD: data/earlynsd/<crop>/<condition>/*.jpg

    Crops: ash_gourd, bitter_gourd, snake_gourd
    Conditions: Healthy, N_deficiency, K_deficiency

    Or flat: data/earlynsd/<crop_condition>/*.jpg
    Or split: data/earlynsd/{train,val,test}/<class>/*.jpg
    """
    samples = []
    if not root.exists():
        return samples

    # Check for train/val/test split directories
    split_dirs = [d for d in root.iterdir()
                  if d.is_dir() and d.name.lower() in ("train", "val", "test", "validation")]

    if split_dirs:
        # Split layout: split/class/images
        for split_dir in sorted(root.iterdir()):
            if not split_dir.is_dir():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                species, condition = _parse_earlynsd_class(class_name)
                for img in class_dir.rglob("*"):
                    if _is_image(img):
                        samples.append((img, species, condition))
    else:
        # Crop/condition or flat layout
        for top_dir in sorted(root.iterdir()):
            if not top_dir.is_dir() or top_dir.name.startswith("."):
                continue

            subdirs = [d for d in top_dir.iterdir() if d.is_dir()]
            if subdirs:
                # Hierarchical: crop/condition
                crop_name = top_dir.name
                for cond_dir in sorted(subdirs):
                    condition = cond_dir.name
                    for img in cond_dir.rglob("*"):
                        if _is_image(img):
                            samples.append((img, crop_name, condition))
            else:
                # Flat: crop_condition folders
                species, condition = _parse_earlynsd_class(top_dir.name)
                for img in top_dir.rglob("*"):
                    if _is_image(img):
                        samples.append((img, species, condition))

    return samples


def _parse_earlynsd_class(class_name: str) -> tuple[str, str]:
    """Parse EarlyNSD class name into (species, condition)."""
    name_lower = class_name.lower().replace("-", "_")

    # Detect condition
    condition: str | None = None
    if "nitrogen" in name_lower or "n_def" in name_lower:
        condition = "Nitrogen Deficiency"
    elif "potassium" in name_lower or "k_def" in name_lower:
        condition = "Potassium Deficiency"
    elif "healthy" in name_lower:
        condition = "Healthy"

    # Detect crop/species
    species = class_name
    for crop in ["ash_gourd", "ashgourd", "bitter_gourd", "bittergourd",
                 "snake_gourd", "snakegourd"]:
        if crop in name_lower:
            species = crop.replace("_", " ").title()
            break

    return species, condition or "Healthy"


# ─── Dataset loader registry ─────────────────────────────────────────────────

DATASET_LOADERS = {
    "herbify": load_herbify,
    "assam": load_assam,
    "medleafx": load_medleafx,
    "cimpd": load_cimpd,
    "simp": load_simp,
    "earlynsd": load_earlynsd,
}


# ─── Federated Dataset ───────────────────────────────────────────────────────

class FederatedBotanicalDataset(Dataset):
    """Unified dataset wrapping all 6 botanical data sources.

    Each sample returns (image_tensor[3, 512, 512], species_id, pathology_id).
    Datasets without pathology labels default to 0 (Healthy).
    """

    def __init__(
        self,
        samples: list[tuple[Path, int, int]],
        transform: A.Compose | None = None,
        image_size: int = 512,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        image_path, species_id, pathology_id = self.samples[idx]

        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
        except (OSError, Image.DecompressionBombError):
            # Return a black image for corrupt files
            img_array = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if self.transform:
            transformed = self.transform(image=img_array)
            image_tensor = transformed["image"]
        else:
            # Fallback: basic resize + tensor conversion
            img = Image.fromarray(img_array).resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            image_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return image_tensor, species_id, pathology_id

    @property
    def species_labels(self) -> list[int]:
        """All species labels for computing class weights / sampling."""
        return [s[1] for s in self.samples]

    @property
    def pathology_labels(self) -> list[int]:
        """All pathology labels for computing class weights / sampling."""
        return [s[2] for s in self.samples]


# ─── Builder: Load, Split, Create DataLoaders ────────────────────────────────

def build_datasets(
    data_root: str | Path,
    datasets: list[str] | None = None,
    image_size: int = 512,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    FederatedBotanicalDataset,
    FederatedBotanicalDataset,
    FederatedBotanicalDataset,
    SpeciesTaxonomy,
]:
    """Load all datasets, build unified taxonomy, stratified split, return train/val/test.

    Returns:
        (train_dataset, val_dataset, test_dataset, species_taxonomy)
    """
    data_root = Path(data_root)
    active_datasets = datasets or list(DATASET_LOADERS.keys())
    taxonomy = SpeciesTaxonomy()

    # Phase 1: Load raw samples from all datasets
    all_samples: list[tuple[Path, int, int]] = []

    for ds_key in active_datasets:
        loader = DATASET_LOADERS.get(ds_key)
        if loader is None:
            print(f"Warning: unknown dataset '{ds_key}', skipping")
            continue

        ds_root = data_root / ds_key
        raw_samples = loader(ds_root)

        for img_path, species_name, raw_pathology in raw_samples:
            species_id = taxonomy.get_or_register(species_name, dataset_source=ds_key)
            pathology_id = map_pathology_label(raw_pathology)
            all_samples.append((img_path, species_id, pathology_id))

        print(f"  [{ds_key}] Loaded {len(raw_samples)} samples")

    if not all_samples:
        print("Warning: no samples loaded from any dataset")
        empty = FederatedBotanicalDataset([], get_val_transforms(image_size), image_size)
        return empty, empty, empty, taxonomy

    print(f"\nTotal: {len(all_samples)} samples, {taxonomy.num_species} species, "
          f"{len(PATHOLOGY_CLASSES)} pathology classes")

    # Phase 2: Stratified split by combined species+pathology label
    stratify_keys = [f"{s}_{p}" for _, s, p in all_samples]

    # Handle rare classes — if any class has < 2 samples, merge into a fallback
    class_counts = Counter(stratify_keys)
    safe_keys = []
    for key in stratify_keys:
        if class_counts[key] < 2:
            safe_keys.append("rare")
        else:
            safe_keys.append(key)

    indices = list(range(len(all_samples)))

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=safe_keys,
        random_state=seed,
    )

    # Second split: val vs test
    val_test_keys = [safe_keys[i] for i in val_test_idx]
    vt_class_counts = Counter(val_test_keys)
    vt_safe_keys = ["rare" if vt_class_counts[k] < 2 else k for k in val_test_keys]

    relative_test = test_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=relative_test,
        stratify=vt_safe_keys,
        random_state=seed,
    )

    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Phase 3: Build dataset objects
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    train_ds = FederatedBotanicalDataset(
        train_samples, get_train_transforms(image_size), image_size
    )
    val_ds = FederatedBotanicalDataset(
        val_samples, get_val_transforms(image_size), image_size
    )
    test_ds = FederatedBotanicalDataset(
        test_samples, get_val_transforms(image_size), image_size
    )

    return train_ds, val_ds, test_ds, taxonomy


def build_weighted_sampler(dataset: FederatedBotanicalDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler to handle class imbalance.

    Weights are computed from species label frequency — rarer species
    get higher sampling probability.
    """
    labels = dataset.species_labels
    class_counts = Counter(labels)
    total = len(labels)

    weights = [total / class_counts[label] for label in labels]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=total,
        replacement=True,
    )


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 8,
    image_size: int = 512,
    datasets: list[str] | None = None,
    seed: int = 42,
    use_weighted_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, SpeciesTaxonomy]:
    """Build train/val/test DataLoaders with optional weighted sampling.

    Returns:
        (train_loader, val_loader, test_loader, species_taxonomy)
    """
    train_ds, val_ds, test_ds, taxonomy = build_datasets(
        data_root=data_root,
        datasets=datasets,
        image_size=image_size,
        seed=seed,
    )

    train_sampler = (
        build_weighted_sampler(train_ds)
        if use_weighted_sampling and len(train_ds) > 0
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and len(train_ds) > 0),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, taxonomy

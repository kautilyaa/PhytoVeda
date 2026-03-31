"""Dataset caching to Drive for Colab runtime restart resilience.

Problem: Downloading ~103K images to Colab SSD takes 10-30 minutes.
         Every runtime restart repeats this.

Solution: After first download + processing, cache the processed dataset
          metadata (file paths, labels, splits) to Drive. On restart,
          rebuild the DataLoader from cached metadata instantly — only
          the raw images need re-downloading (or can be cached too).

Cache layers:
    1. Download cache — track which datasets are already extracted on SSD
    2. Taxonomy cache — species/pathology label mappings
    3. Split cache — train/val/test split indices (reproducible across restarts)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CacheMetadata:
    """Metadata about a cached artifact."""

    cache_key: str
    created_at: float  # Unix timestamp
    source_hash: str  # Hash of inputs that generated this cache
    item_count: int
    size_bytes: int = 0
    extra: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> CacheMetadata:
        return cls(**data)


class DatasetCache:
    """Cache dataset metadata and split indices to Drive.

    Stores lightweight JSON files — not the images themselves. This allows
    instant DataLoader reconstruction on runtime restart as long as images
    are available on SSD.

    Usage:
        cache = DatasetCache(cache_dir=dm.drive_base / "cache")

        # Save split indices after first build
        cache.save_splits(train_indices, val_indices, test_indices, split_hash="...")

        # On restart — check if splits are cached
        if cache.has_splits(split_hash):
            train_idx, val_idx, test_idx = cache.load_splits()
        else:
            # Rebuild from scratch
            ...

        # Cache taxonomy mapping
        cache.save_taxonomy(species_map, pathology_map)
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ─── Split Indices ───────────────────────────────────────────────────

    def _splits_path(self) -> Path:
        return self.cache_dir / "splits.json"

    def _splits_meta_path(self) -> Path:
        return self.cache_dir / "splits_meta.json"

    def save_splits(
        self,
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        split_hash: str = "",
        dataset_sizes: dict[str, int] | None = None,
    ) -> Path:
        """Cache train/val/test split indices.

        Args:
            train_indices: Training set indices.
            val_indices: Validation set indices.
            test_indices: Test set indices.
            split_hash: Hash identifying the split configuration.
            dataset_sizes: Optional dict of dataset name → image count.

        Returns:
            Path to the saved splits file.
        """
        data = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        path = self._splits_path()
        path.write_text(json.dumps(data), encoding="utf-8")

        # Save metadata
        meta = CacheMetadata(
            cache_key="splits",
            created_at=time.time(),
            source_hash=split_hash,
            item_count=len(train_indices) + len(val_indices) + len(test_indices),
            size_bytes=path.stat().st_size,
            extra={"dataset_sizes": dataset_sizes} if dataset_sizes else None,
        )
        self._splits_meta_path().write_text(
            json.dumps(meta.to_dict()), encoding="utf-8"
        )
        return path

    def has_splits(self, split_hash: str = "") -> bool:
        """Check if cached splits exist and match the expected hash.

        Args:
            split_hash: If provided, only return True if hash matches.
        """
        if not self._splits_path().exists():
            return False
        if not split_hash:
            return True
        # Check hash
        if self._splits_meta_path().exists():
            meta = json.loads(self._splits_meta_path().read_text(encoding="utf-8"))
            return meta.get("source_hash", "") == split_hash
        return False

    def load_splits(self) -> tuple[list[int], list[int], list[int]]:
        """Load cached split indices.

        Returns:
            (train_indices, val_indices, test_indices)

        Raises:
            FileNotFoundError: If no cached splits exist.
        """
        path = self._splits_path()
        if not path.exists():
            raise FileNotFoundError("No cached splits found")

        data = json.loads(path.read_text(encoding="utf-8"))
        return data["train"], data["val"], data["test"]

    # ─── Taxonomy Mapping ────────────────────────────────────────────────

    def _taxonomy_path(self) -> Path:
        return self.cache_dir / "taxonomy.json"

    def save_taxonomy(
        self,
        species_map: dict[str, int],
        pathology_map: dict[str, int],
    ) -> Path:
        """Cache species and pathology label-to-ID mappings.

        Args:
            species_map: Species name → numeric ID.
            pathology_map: Pathology label → numeric ID.

        Returns:
            Path to the saved taxonomy file.
        """
        data = {
            "species": species_map,
            "pathology": pathology_map,
            "num_species": len(species_map),
            "num_pathology": len(pathology_map),
        }
        path = self._taxonomy_path()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def has_taxonomy(self) -> bool:
        """Check if cached taxonomy exists."""
        return self._taxonomy_path().exists()

    def load_taxonomy(self) -> dict:
        """Load cached taxonomy mapping.

        Returns:
            Dict with 'species', 'pathology', 'num_species', 'num_pathology' keys.

        Raises:
            FileNotFoundError: If no cached taxonomy exists.
        """
        path = self._taxonomy_path()
        if not path.exists():
            raise FileNotFoundError("No cached taxonomy found")
        return json.loads(path.read_text(encoding="utf-8"))

    # ─── Download Status Tracking ────────────────────────────────────────

    def _downloads_path(self) -> Path:
        return self.cache_dir / "download_status.json"

    def mark_downloaded(self, dataset_name: str, image_count: int, path: str) -> None:
        """Mark a dataset as successfully downloaded.

        Args:
            dataset_name: Name of the dataset (e.g., "herbify").
            image_count: Number of images extracted.
            path: Path where dataset was extracted.
        """
        status = self._load_download_status()
        status[dataset_name] = {
            "image_count": image_count,
            "path": path,
            "downloaded_at": time.time(),
        }
        self._downloads_path().write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )

    def is_downloaded(self, dataset_name: str, min_images: int = 0) -> bool:
        """Check if a dataset is marked as downloaded.

        Args:
            dataset_name: Name of the dataset.
            min_images: Minimum image count to consider valid.
        """
        status = self._load_download_status()
        if dataset_name not in status:
            return False
        entry = status[dataset_name]
        if min_images > 0 and entry.get("image_count", 0) < min_images:
            return False
        # Also check the path still exists on SSD
        return Path(entry["path"]).exists()

    def download_status(self) -> dict[str, dict]:
        """Get download status for all datasets."""
        return self._load_download_status()

    def _load_download_status(self) -> dict[str, dict]:
        path = self._downloads_path()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    # ─── Training History ────────────────────────────────────────────────

    def _history_path(self) -> Path:
        return self.cache_dir / "training_history.json"

    def save_history(self, history: dict[str, list[float]]) -> Path:
        """Cache training history (loss curves, F1 per epoch).

        Args:
            history: Dict of metric name → list of per-epoch values.

        Returns:
            Path to saved history file.
        """
        path = self._history_path()
        path.write_text(json.dumps(history), encoding="utf-8")
        return path

    def has_history(self) -> bool:
        """Check if cached training history exists."""
        return self._history_path().exists()

    def load_history(self) -> dict[str, list[float]]:
        """Load cached training history.

        Returns:
            Dict of metric name → list of per-epoch values.

        Raises:
            FileNotFoundError: If no cached history exists.
        """
        path = self._history_path()
        if not path.exists():
            raise FileNotFoundError("No cached training history found")
        return json.loads(path.read_text(encoding="utf-8"))

    # ─── Utilities ───────────────────────────────────────────────────────

    def clear(self) -> int:
        """Clear all cached data.

        Returns:
            Number of files removed.
        """
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file() and f.suffix == ".json":
                f.unlink()
                count += 1
        return count

    def summary(self) -> str:
        """Human-readable cache status summary."""
        lines = [f"Dataset Cache: {self.cache_dir}", "-" * 50]

        # Splits
        if self.has_splits():
            train, val, test = self.load_splits()
            lines.append(f"  Splits:    {len(train)} train / {len(val)} val / {len(test)} test")
        else:
            lines.append("  Splits:    Not cached")

        # Taxonomy
        if self.has_taxonomy():
            tax = self.load_taxonomy()
            lines.append(
                f"  Taxonomy:  {tax['num_species']} species, "
                f"{tax['num_pathology']} pathology"
            )
        else:
            lines.append("  Taxonomy:  Not cached")

        # Downloads
        status = self.download_status()
        if status:
            lines.append(f"  Downloads: {len(status)} datasets tracked")
            for name, info in status.items():
                exists = "SSD" if Path(info["path"]).exists() else "MISSING"
                lines.append(f"    {name:15s} {info['image_count']:>6} images [{exists}]")
        else:
            lines.append("  Downloads: No status")

        # History
        if self.has_history():
            hist = self.load_history()
            epochs = len(next(iter(hist.values()), []))
            lines.append(f"  History:   {epochs} epochs")
        else:
            lines.append("  History:   Not cached")

        return "\n".join(lines)


def compute_split_hash(
    dataset_names: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> str:
    """Compute a deterministic hash for a split configuration.

    Same inputs always produce the same hash, so we can detect when
    splits need to be regenerated.
    """
    key = f"{sorted(dataset_names)}_{train_ratio}_{val_ratio}_{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

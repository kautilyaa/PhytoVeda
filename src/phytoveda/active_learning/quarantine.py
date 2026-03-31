"""Quarantine pipeline: route uncertain images to local storage and GCS for oracle labeling.

Flagged images are stored with JSON metadata including timestamp, model version,
uncertainty scores, and top-K predictions. Supports local-first workflow with
optional GCS sync.

Local layout:
    quarantine_dir/
        <image_hash>/
            image.jpg          # Copy of the original image
            metadata.json      # QuarantineEntry as JSON
        manifest.json          # Index of all quarantined images
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from phytoveda.active_learning.uncertainty import UncertaintyScore


@dataclass
class QuarantineEntry:
    """Metadata for a quarantined uncertain image."""

    image_path: str
    quarantine_id: str
    timestamp: str
    model_version: str
    uncertainty: dict  # Serialized UncertaintyScore
    top_k_species: list[tuple[str, float]]
    top_k_pathology: list[tuple[str, float]]
    labeled: bool = False
    oracle_species: str = ""
    oracle_pathology: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QuarantineEntry:
        return cls(**data)


def _hash_path(image_path: str) -> str:
    """Generate a short deterministic hash for an image path."""
    return sha256(image_path.encode()).hexdigest()[:16]


class QuarantineManager:
    """Manage local quarantine directory for active learning pipeline.

    Images flagged as uncertain are copied to the quarantine directory with
    metadata. Once labeled by an oracle (LLM or human), they can be exported
    for incremental retraining.
    """

    def __init__(
        self,
        local_dir: str | Path = "data/quarantine",
        gcs_bucket: str | None = None,
        model_version: str = "unknown",
    ) -> None:
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_bucket = gcs_bucket
        self.model_version = model_version
        self._manifest: list[QuarantineEntry] = []
        self._load_manifest()

    def _manifest_path(self) -> Path:
        return self.local_dir / "manifest.json"

    def _load_manifest(self) -> None:
        """Load existing manifest from disk."""
        manifest_file = self._manifest_path()
        if manifest_file.exists():
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            self._manifest = [QuarantineEntry.from_dict(e) for e in data]

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        data = [e.to_dict() for e in self._manifest]
        self._manifest_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def quarantine(
        self,
        image_path: str | Path,
        uncertainty: UncertaintyScore,
        top_k_species: list[tuple[str, float]],
        top_k_pathology: list[tuple[str, float]],
    ) -> QuarantineEntry:
        """Copy an uncertain image to quarantine with metadata.

        Args:
            image_path: Path to the original image file.
            uncertainty: Uncertainty scores from the sampler.
            top_k_species: Top-K species predictions as (name, prob) pairs.
            top_k_pathology: Top-K pathology predictions as (name, prob) pairs.

        Returns:
            The created QuarantineEntry.
        """
        image_path = Path(image_path)
        qid = _hash_path(str(image_path))

        # Create quarantine subdirectory
        q_dir = self.local_dir / qid
        q_dir.mkdir(parents=True, exist_ok=True)

        # Copy image
        dest_image = q_dir / f"image{image_path.suffix}"
        if image_path.exists():
            shutil.copy2(image_path, dest_image)

        entry = QuarantineEntry(
            image_path=str(image_path),
            quarantine_id=qid,
            timestamp=datetime.now(UTC).isoformat(),
            model_version=self.model_version,
            uncertainty={
                "least_confidence": uncertainty.least_confidence,
                "margin": uncertainty.margin,
                "entropy": uncertainty.entropy,
                "combined": uncertainty.combined,
                "is_uncertain": uncertainty.is_uncertain,
            },
            top_k_species=top_k_species,
            top_k_pathology=top_k_pathology,
        )

        # Save metadata JSON
        meta_path = q_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self._manifest.append(entry)
        self._save_manifest()

        return entry

    def scan(self) -> list[QuarantineEntry]:
        """Return all quarantined entries."""
        return list(self._manifest)

    def pending(self) -> list[QuarantineEntry]:
        """Return unlabeled quarantine entries."""
        return [e for e in self._manifest if not e.labeled]

    def labeled_entries(self) -> list[QuarantineEntry]:
        """Return entries that have been labeled by an oracle."""
        return [e for e in self._manifest if e.labeled]

    def mark_labeled(
        self,
        quarantine_id: str,
        species_label: str,
        pathology_label: str,
    ) -> bool:
        """Mark a quarantined image as labeled by an oracle.

        Returns True if the entry was found and updated.
        """
        for entry in self._manifest:
            if entry.quarantine_id == quarantine_id:
                entry.labeled = True
                entry.oracle_species = species_label
                entry.oracle_pathology = pathology_label
                self._save_manifest()

                # Also update the individual metadata file
                q_dir = self.local_dir / quarantine_id
                meta_path = q_dir / "metadata.json"
                if meta_path.exists():
                    meta_path.write_text(
                        json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return True
        return False

    def export_for_retraining(self) -> list[tuple[Path, str, str]]:
        """Export labeled quarantine images for incremental retraining.

        Returns:
            List of (image_path, species_label, pathology_label) tuples
            ready to be fed into the data pipeline.
        """
        exports: list[tuple[Path, str, str]] = []
        for entry in self.labeled_entries():
            q_dir = self.local_dir / entry.quarantine_id
            # Find the copied image
            images = list(q_dir.glob("image.*"))
            if images:
                exports.append((images[0], entry.oracle_species, entry.oracle_pathology))
        return exports

    @property
    def total_count(self) -> int:
        return len(self._manifest)

    @property
    def pending_count(self) -> int:
        return len(self.pending())

    @property
    def labeled_count(self) -> int:
        return len(self.labeled_entries())

    def summary(self) -> str:
        """Human-readable quarantine summary."""
        return (
            f"Quarantine: {self.total_count} total, "
            f"{self.pending_count} pending, "
            f"{self.labeled_count} labeled"
        )

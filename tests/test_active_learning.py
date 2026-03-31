"""Tests for the active learning pipeline: quarantine and oracle.

Uncertainty sampler tests are in test_uncertainty.py (already exists).
These tests cover quarantine management and oracle labeling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phytoveda.active_learning.oracle import (
    HumanExpertQueue,
    OracleLabel,
    OraclePipeline,
    OracleSource,
)
from phytoveda.active_learning.quarantine import QuarantineEntry, QuarantineManager
from phytoveda.active_learning.uncertainty import UncertaintySampler, UncertaintyScore


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(path, format="JPEG")


def _make_uncertainty(uncertain: bool = True) -> UncertaintyScore:
    return UncertaintyScore(
        least_confidence=0.7 if uncertain else 0.1,
        margin=0.05 if uncertain else 0.8,
        entropy=3.0 if uncertain else 0.5,
        combined=0.6 if uncertain else 0.1,
        is_uncertain=uncertain,
    )


def _make_top_k() -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    species = [("Azadirachta indica", 0.4), ("Terminalia chebula", 0.35), ("Moringa oleifera", 0.15)]
    pathology = [("Bacterial Spot", 0.5), ("Healthy", 0.3), ("Yellow Leaf Disease", 0.2)]
    return species, pathology


# ─── QuarantineManager Tests ───────────────────────────────────────────────


class TestQuarantineManager:
    def test_quarantine_image(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir, model_version="v0.1")
        species, pathology = _make_top_k()
        entry = mgr.quarantine(img_path, _make_uncertainty(), species, pathology)

        assert entry.quarantine_id
        assert entry.model_version == "v0.1"
        assert entry.labeled is False
        assert mgr.total_count == 1
        assert mgr.pending_count == 1

    def test_image_copied_to_quarantine(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        entry = mgr.quarantine(img_path, _make_uncertainty(), species, pathology)

        # Image should be copied
        copied = list((q_dir / entry.quarantine_id).glob("image.*"))
        assert len(copied) == 1

    def test_metadata_json_saved(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        entry = mgr.quarantine(img_path, _make_uncertainty(), species, pathology)

        meta_path = q_dir / entry.quarantine_id / "metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["quarantine_id"] == entry.quarantine_id
        assert data["uncertainty"]["is_uncertain"] is True

    def test_manifest_persists(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr1 = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        mgr1.quarantine(img_path, _make_uncertainty(), species, pathology)

        # New manager should load the manifest
        mgr2 = QuarantineManager(local_dir=q_dir)
        assert mgr2.total_count == 1

    def test_mark_labeled(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        entry = mgr.quarantine(img_path, _make_uncertainty(), species, pathology)

        result = mgr.mark_labeled(entry.quarantine_id, "Azadirachta indica", "Bacterial Spot")
        assert result is True
        assert mgr.pending_count == 0
        assert mgr.labeled_count == 1

    def test_mark_labeled_unknown_id(self, tmp_path: Path) -> None:
        mgr = QuarantineManager(local_dir=tmp_path / "quarantine")
        assert mgr.mark_labeled("nonexistent", "sp", "path") is False

    def test_export_for_retraining(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        entry = mgr.quarantine(img_path, _make_uncertainty(), species, pathology)
        mgr.mark_labeled(entry.quarantine_id, "Neem", "Healthy")

        exports = mgr.export_for_retraining()
        assert len(exports) == 1
        path, sp_label, path_label = exports[0]
        assert path.exists()
        assert sp_label == "Neem"
        assert path_label == "Healthy"

    def test_export_empty_when_no_labels(self, tmp_path: Path) -> None:
        img_path = tmp_path / "leaf.jpg"
        _make_image(img_path)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        mgr.quarantine(img_path, _make_uncertainty(), species, pathology)

        assert mgr.export_for_retraining() == []

    def test_multiple_quarantine_entries(self, tmp_path: Path) -> None:
        q_dir = tmp_path / "quarantine"
        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()

        for i in range(5):
            img = tmp_path / f"leaf_{i}.jpg"
            _make_image(img)
            mgr.quarantine(img, _make_uncertainty(), species, pathology)

        assert mgr.total_count == 5
        assert mgr.pending_count == 5

    def test_summary(self, tmp_path: Path) -> None:
        mgr = QuarantineManager(local_dir=tmp_path / "q")
        s = mgr.summary()
        assert "0 total" in s
        assert "0 pending" in s


# ─── QuarantineEntry Serialization Tests ────────────────────────────────────


class TestQuarantineEntry:
    def test_round_trip(self) -> None:
        entry = QuarantineEntry(
            image_path="/path/to/img.jpg",
            quarantine_id="abc123",
            timestamp="2026-03-30T12:00:00",
            model_version="v0.1",
            uncertainty={"least_confidence": 0.7, "margin": 0.05, "entropy": 3.0,
                         "combined": 0.6, "is_uncertain": True},
            top_k_species=[("Neem", 0.4)],
            top_k_pathology=[("Healthy", 0.5)],
        )
        d = entry.to_dict()
        restored = QuarantineEntry.from_dict(d)
        assert restored.quarantine_id == "abc123"
        assert restored.model_version == "v0.1"


# ─── HumanExpertQueue Tests ────────────────────────────────────────────────


class TestHumanExpertQueue:
    def _make_entry(self) -> QuarantineEntry:
        return QuarantineEntry(
            image_path="/fake/path.jpg",
            quarantine_id="test_q1",
            timestamp="2026-03-30T12:00:00",
            model_version="v0.1",
            uncertainty={"least_confidence": 0.7, "margin": 0.05, "entropy": 3.0,
                         "combined": 0.6, "is_uncertain": True},
            top_k_species=[("Neem", 0.4)],
            top_k_pathology=[("Healthy", 0.5)],
        )

    def test_enqueue(self, tmp_path: Path) -> None:
        queue = HumanExpertQueue(queue_dir=tmp_path / "expert")
        queue.enqueue(self._make_entry())
        assert queue.pending_count == 1

    def test_no_duplicates(self, tmp_path: Path) -> None:
        queue = HumanExpertQueue(queue_dir=tmp_path / "expert")
        entry = self._make_entry()
        queue.enqueue(entry)
        queue.enqueue(entry)
        assert queue.pending_count == 1

    def test_submit_label(self, tmp_path: Path) -> None:
        queue = HumanExpertQueue(queue_dir=tmp_path / "expert")
        queue.enqueue(self._make_entry())

        label = queue.submit_label("test_q1", "Azadirachta indica", "Bacterial Spot", "Clear lesions")
        assert label is not None
        assert label.source == OracleSource.HUMAN_EXPERT
        assert label.confidence == 1.0
        assert label.species_label == "Azadirachta indica"
        assert queue.pending_count == 0
        assert queue.labeled_count == 1

    def test_submit_label_unknown_id(self, tmp_path: Path) -> None:
        queue = HumanExpertQueue(queue_dir=tmp_path / "expert")
        assert queue.submit_label("nope", "sp", "path") is None

    def test_persistence(self, tmp_path: Path) -> None:
        q_dir = tmp_path / "expert"
        queue1 = HumanExpertQueue(queue_dir=q_dir)
        queue1.enqueue(self._make_entry())
        queue1.submit_label("test_q1", "Neem", "Healthy")

        # New instance should load state
        queue2 = HumanExpertQueue(queue_dir=q_dir)
        assert queue2.pending_count == 0
        assert queue2.labeled_count == 1
        assert queue2.all_labels[0].species_label == "Neem"


# ─── OracleLabel Serialization Tests ────────────────────────────────────────


class TestOracleLabel:
    def test_round_trip(self) -> None:
        label = OracleLabel(
            quarantine_id="q1",
            image_path="/img.jpg",
            species_label="Neem",
            pathology_label="Healthy",
            source=OracleSource.LLM,
            confidence=0.95,
            notes="High confidence",
        )
        d = label.to_dict()
        assert d["source"] == "llm"
        restored = OracleLabel.from_dict(d)
        assert restored.source == OracleSource.LLM
        assert restored.confidence == 0.95


# ─── OraclePipeline Tests ──────────────────────────────────────────────────


class TestOraclePipeline:
    def test_no_llm_routes_to_human(self, tmp_path: Path) -> None:
        """Without LLM oracle, all entries go to human queue."""
        img = tmp_path / "leaf.jpg"
        _make_image(img)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        mgr.quarantine(img, _make_uncertainty(), species, pathology)

        expert = HumanExpertQueue(queue_dir=tmp_path / "expert")
        pipeline = OraclePipeline(quarantine=mgr, llm_oracle=None, expert_queue=expert)

        stats = pipeline.process_pending()
        assert stats["human_routed"] == 1
        assert stats["llm_labeled"] == 0
        assert expert.pending_count == 1

    def test_apply_human_labels(self, tmp_path: Path) -> None:
        img = tmp_path / "leaf.jpg"
        _make_image(img)
        q_dir = tmp_path / "quarantine"

        mgr = QuarantineManager(local_dir=q_dir)
        species, pathology = _make_top_k()
        entry = mgr.quarantine(img, _make_uncertainty(), species, pathology)

        expert = HumanExpertQueue(queue_dir=tmp_path / "expert")
        expert.enqueue(entry)
        expert.submit_label(entry.quarantine_id, "Neem", "Healthy")

        pipeline = OraclePipeline(quarantine=mgr, expert_queue=expert)
        applied = pipeline.apply_human_labels()
        assert applied == 1
        assert mgr.labeled_count == 1


# ─── Integration: Uncertainty -> Quarantine Flow ────────────────────────────


class TestUncertaintyToQuarantine:
    def test_end_to_end_flow(self, tmp_path: Path) -> None:
        """Simulate: model produces uncertain prediction -> quarantine -> label -> export."""
        import torch

        # 1. Model produces uncertain logits
        sampler = UncertaintySampler(confidence_threshold=0.5, margin_threshold=0.1)
        logits = torch.tensor([[2.0, 1.95, 0.1, 0.05]])  # Very close top-2
        scores = sampler.score(logits)
        assert scores[0].is_uncertain

        # 2. Quarantine the image
        img = tmp_path / "uncertain_leaf.jpg"
        _make_image(img)
        q_dir = tmp_path / "quarantine"
        mgr = QuarantineManager(local_dir=q_dir, model_version="v0.1")

        species_top_k = [("Neem", 0.4), ("Haritaki", 0.38)]
        pathology_top_k = [("Bacterial Spot", 0.5), ("Healthy", 0.3)]
        entry = mgr.quarantine(img, scores[0], species_top_k, pathology_top_k)

        assert mgr.pending_count == 1

        # 3. Oracle labels it
        mgr.mark_labeled(entry.quarantine_id, "Azadirachta indica", "Bacterial Spot")
        assert mgr.labeled_count == 1

        # 4. Export for retraining
        exports = mgr.export_for_retraining()
        assert len(exports) == 1
        path, sp, pa = exports[0]
        assert sp == "Azadirachta indica"
        assert pa == "Bacterial Spot"
        assert path.exists()

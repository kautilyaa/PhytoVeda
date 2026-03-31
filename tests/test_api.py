"""Tests for the PhytoVeda API server and inference pipeline.

Uses a lightweight ViT backbone (vit_tiny_patch16_224) and FastAPI TestClient
so tests run without GPU or large model downloads.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from phytoveda.api.inference import (
    InferencePipeline,
    PredictionResult,
    build_inference_transform,
)
from phytoveda.data.taxonomy import ID_TO_PATHOLOGY, SpeciesTaxonomy
from phytoveda.models.cmtl import HierViTCMTL
from phytoveda.rag.report_generator import ReportGenerator
from phytoveda.vrikshayurveda.mapper import Dosha


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_image(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a random RGB test image."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_model(num_species: int = 10, num_pathologies: int = 8) -> HierViTCMTL:
    """Create a lightweight test model (vit_tiny_patch16_224, untrained)."""
    return HierViTCMTL(
        num_species=num_species,
        num_pathologies=num_pathologies,
        backbone_name="vit_tiny_patch16_224",
        pretrained=False,
        image_size=224,
        species_hidden_dim=64,
        pathology_hidden_dim=32,
        dropout=0.0,
    )


def _make_pipeline(
    num_species: int = 10,
    num_pathologies: int = 8,
    with_report_gen: bool = False,
) -> InferencePipeline:
    """Create a test inference pipeline."""
    model = _make_model(num_species, num_pathologies)
    taxonomy = SpeciesTaxonomy()
    report_gen = ReportGenerator() if with_report_gen else None
    return InferencePipeline(
        model=model,
        taxonomy=taxonomy,
        device=torch.device("cpu"),
        image_size=224,
        report_generator=report_gen,
    )


def _save_checkpoint(path: Path, num_species: int = 10, num_pathologies: int = 8) -> Path:
    """Save a lightweight model checkpoint for testing."""
    model = _make_model(num_species, num_pathologies)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_species": num_species,
            "num_pathologies": num_pathologies,
            "backbone": "vit_tiny_patch16_224",
            "image_size": 224,
            "version": "v0.1-test",
        },
        "metrics": {"species_f1": 0.85, "pathology_f1": 0.90},
        "epoch": 10,
    }
    ckpt_path = path / "test_model.pt"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def _image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ─── INFERENCE_TRANSFORM Tests ─────────────────────────────────────────────


class TestInferenceTransform:
    def test_output_shape_512(self) -> None:
        img = _make_image((300, 400))
        tensor = build_inference_transform(512)(img)
        assert tensor.shape == (3, 512, 512)

    def test_output_shape_224(self) -> None:
        img = _make_image((300, 400))
        tensor = build_inference_transform(224)(img)
        assert tensor.shape == (3, 224, 224)

    def test_normalized_range(self) -> None:
        img = _make_image()
        tensor = build_inference_transform(224)(img)
        # After ImageNet normalization, values are roughly in [-3, 3]
        assert tensor.min() > -5.0
        assert tensor.max() < 5.0


# ─── InferencePipeline Tests ──────────────────────────────────────────────


class TestInferencePipeline:
    def test_predict_returns_result(self) -> None:
        pipeline = _make_pipeline()
        image = _make_image()
        result = pipeline.predict(image)
        assert isinstance(result, PredictionResult)

    def test_predict_species_fields(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image())
        assert isinstance(result.species_name, str)
        assert 0.0 <= result.species_confidence <= 1.0
        assert len(result.top_k_species) > 0

    def test_predict_pathology_fields(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image())
        assert result.pathology_label in ID_TO_PATHOLOGY.values()
        assert 0.0 <= result.pathology_confidence <= 1.0
        assert len(result.top_k_pathology) > 0

    def test_predict_dosha_field(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image())
        assert result.dosha.dosha in Dosha
        assert isinstance(result.dosha.treatments, list)

    def test_predict_uncertainty_field(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image())
        assert isinstance(result.uncertainty.least_confidence, float)
        assert isinstance(result.uncertainty.is_uncertain, bool)

    def test_predict_top_k(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image(), top_k=3)
        assert len(result.top_k_species) == 3
        assert len(result.top_k_pathology) == 3

    def test_predict_top_k_clamped(self) -> None:
        """top_k > num_classes returns all classes."""
        pipeline = _make_pipeline(num_species=5, num_pathologies=4)
        result = pipeline.predict(_make_image(), top_k=100)
        assert len(result.top_k_species) == 5
        assert len(result.top_k_pathology) == 4

    def test_predict_no_report_by_default(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image())
        assert result.report is None

    def test_predict_offline_report(self) -> None:
        pipeline = _make_pipeline(with_report_gen=True)
        # generate_report=True but no Gemini key → falls back to offline
        result = pipeline.predict(_make_image(), generate_report=True)
        assert result.report is not None
        assert result.report.species_name == result.species_name
        assert result.report.pathology_diagnosis == result.pathology_label

    def test_to_dict_structure(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.predict(_make_image())
        d = result.to_dict()
        assert "species_name" in d
        assert "pathology_label" in d
        assert "dosha" in d
        assert "uncertainty" in d
        assert d["dosha"]["dosha"] in [ds.value for ds in Dosha]

    def test_to_dict_with_report(self) -> None:
        pipeline = _make_pipeline(with_report_gen=True)
        result = pipeline.predict(_make_image(), generate_report=True)
        d = result.to_dict()
        assert "report" in d
        assert "procurement_quality" in d["report"]

    def test_preprocess_output(self) -> None:
        pipeline = _make_pipeline()
        img = _make_image()
        tensor = pipeline.preprocess(img)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.device == pipeline.device

    def test_different_image_sizes(self) -> None:
        pipeline = _make_pipeline()
        for size in [(100, 100), (800, 600), (50, 1000)]:
            result = pipeline.predict(_make_image(size))
            assert isinstance(result, PredictionResult)


# ─── Checkpoint Loading Tests ──────────────────────────────────────────────


class TestFromCheckpoint:
    def test_load_checkpoint(self, tmp_path: Path) -> None:
        ckpt = _save_checkpoint(tmp_path)
        pipeline = InferencePipeline.from_checkpoint(
            ckpt,
            device=torch.device("cpu"),
            species_hidden_dim=64,
            pathology_hidden_dim=32,
            dropout=0.0,
        )
        assert pipeline.model is not None
        assert pipeline.image_size == 224
        result = pipeline.predict(_make_image())
        assert isinstance(result, PredictionResult)

    def test_checkpoint_config_used(self, tmp_path: Path) -> None:
        ckpt = _save_checkpoint(tmp_path, num_species=15, num_pathologies=6)
        pipeline = InferencePipeline.from_checkpoint(
            ckpt,
            device=torch.device("cpu"),
            species_hidden_dim=64,
            pathology_hidden_dim=32,
            dropout=0.0,
        )
        # Model should have been constructed with checkpoint's config
        # head is nn.Sequential: [Linear, GELU, Dropout, Linear(out)]
        assert pipeline.model.species_head.head[3].out_features == 15
        assert pipeline.model.pathology_head.head[3].out_features == 6

    def test_checkpoint_override_kwargs(self, tmp_path: Path) -> None:
        ckpt = _save_checkpoint(tmp_path, num_species=10)
        # Override num_species in kwargs — should take precedence
        pipeline = InferencePipeline.from_checkpoint(
            ckpt,
            device=torch.device("cpu"),
            num_species=10,
            species_hidden_dim=64,
            pathology_hidden_dim=32,
            dropout=0.0,
        )
        assert pipeline.model.species_head.head[3].out_features == 10


# ─── FastAPI Endpoint Tests ────────────────────────────────────────────────


class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def _setup_app(self) -> None:
        """Inject a test pipeline into app state before each test."""
        from phytoveda.api import server

        pipeline = _make_pipeline()
        server._state["pipeline"] = pipeline
        server._state["model_version"] = "v0.1-test"
        server._state["checkpoint_path"] = "/fake/checkpoint.pt"
        server._state["metrics"] = {"species_f1": 0.85}
        server._state["start_time"] = 1000000000.0
        yield
        server._state["pipeline"] = None

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from phytoveda.api.server import app
        return TestClient(app)

    def test_health_with_model(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"

    def test_health_no_model(self, client) -> None:
        from phytoveda.api import server
        server._state["pipeline"] = None
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "no_model"
        assert data["model_loaded"] is False

    def test_model_version(self, client) -> None:
        resp = client.get("/model/version")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "v0.1-test"
        assert data["metrics"]["species_f1"] == 0.85

    def test_identify_jpeg(self, client) -> None:
        img_bytes = _image_to_bytes(_make_image(), "JPEG")
        resp = client.post(
            "/identify",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "species_name" in data
        assert "pathology_label" in data
        assert "dosha" in data
        assert "uncertainty" in data
        assert "inference_time_ms" in data

    def test_identify_png(self, client) -> None:
        img_bytes = _image_to_bytes(_make_image(), "PNG")
        resp = client.post(
            "/identify",
            files={"file": ("leaf.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "species_name" in data

    def test_identify_top_k(self, client) -> None:
        img_bytes = _image_to_bytes(_make_image(), "JPEG")
        resp = client.post(
            "/identify?top_k=3",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        data = resp.json()
        assert len(data["top_k_species"]) == 3

    def test_identify_unsupported_type(self, client) -> None:
        resp = client.post(
            "/identify",
            files={"file": ("doc.pdf", b"fake", "application/pdf")},
        )
        assert resp.status_code == 415

    def test_identify_invalid_image(self, client) -> None:
        resp = client.post(
            "/identify",
            files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_identify_no_model(self, client) -> None:
        from phytoveda.api import server
        server._state["pipeline"] = None
        img_bytes = _image_to_bytes(_make_image(), "JPEG")
        resp = client.post(
            "/identify",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 503

    def test_identify_response_shape(self, client) -> None:
        """Verify the full response shape matches the API contract."""
        img_bytes = _image_to_bytes(_make_image(), "JPEG")
        resp = client.post(
            "/identify",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        data = resp.json()

        # Top-level keys
        assert "species_name" in data
        assert "species_confidence" in data
        assert "top_k_species" in data
        assert "pathology_label" in data
        assert "pathology_confidence" in data
        assert "top_k_pathology" in data
        assert "dosha" in data
        assert "uncertainty" in data

        # Dosha shape
        dosha = data["dosha"]
        assert "dosha" in dosha
        assert "treatments" in dosha
        assert "cv_features" in dosha

        # Uncertainty shape
        unc = data["uncertainty"]
        assert "is_uncertain" in unc
        assert "least_confidence" in unc

        # Top-K items shape
        for item in data["top_k_species"]:
            assert "name" in item
            assert "confidence" in item

    def test_identify_with_report(self, client) -> None:
        """Report generation falls back to offline when no Gemini key."""
        from phytoveda.api import server
        pipeline = _make_pipeline(with_report_gen=True)
        server._state["pipeline"] = pipeline
        server._state["start_time"] = 1000000000.0

        img_bytes = _image_to_bytes(_make_image(), "JPEG")
        resp = client.post(
            "/identify?generate_report=true",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        data = resp.json()
        assert "report" in data
        assert "procurement_quality" in data["report"]

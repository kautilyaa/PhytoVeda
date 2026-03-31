"""Comprehensive tests for the data pipeline.

Tests cover:
    - SpeciesTaxonomy: registration, deduplication, name variants
    - Pathology mapping: raw labels -> unified IDs
    - Per-dataset loaders: directory layout parsing for all 6 datasets
    - FederatedBotanicalDataset: __getitem__, corrupt image handling
    - build_datasets: stratified splitting, taxonomy integration
    - build_weighted_sampler: class weighting correctness
    - Augmentation: transform shape and dtype
    - Preprocessing: image resize, corrupt handling
    - Download validation: image counting, validation logic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from phytoveda.data.augmentation import get_train_transforms, get_val_transforms
from phytoveda.data.datasets import (
    FederatedBotanicalDataset,
    build_datasets,
    build_weighted_sampler,
    load_assam,
    load_cimpd,
    load_earlynsd,
    load_herbify,
    load_medleafx,
    load_simp,
)
from phytoveda.data.download import count_images, validate_dataset
from phytoveda.data.preprocessing import preprocess_image, validate_image
from phytoveda.data.taxonomy import (
    PATHOLOGY_CLASSES,
    PATHOLOGY_TO_ID,
    SpeciesTaxonomy,
    map_pathology_label,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a minimal valid JPEG image at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(path, format="JPEG")


def _make_corrupt_file(path: Path) -> None:
    """Write a non-image file with an image extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an image at all")


# ─── Taxonomy Tests ─────────────────────────────────────────────────────────


class TestSpeciesTaxonomy:
    """Tests for SpeciesTaxonomy registration and deduplication."""

    def test_canonical_species_registered(self) -> None:
        """All 10 canonical species should be pre-registered."""
        tax = SpeciesTaxonomy()
        assert tax.num_species == 10

    def test_canonical_lookup_by_scientific_name(self) -> None:
        tax = SpeciesTaxonomy()
        sid = tax.get_id("Azadirachta indica")
        assert sid is not None
        info = tax.get_info(sid)
        assert info is not None
        assert info.common_name == "Neem"

    def test_canonical_lookup_by_common_name(self) -> None:
        tax = SpeciesTaxonomy()
        assert tax.get_id("Neem") is not None
        assert tax.get_id("Neem") == tax.get_id("Azadirachta indica")

    def test_canonical_lookup_by_sanskrit_name(self) -> None:
        tax = SpeciesTaxonomy()
        assert tax.get_id("Nimba") == tax.get_id("Azadirachta indica")

    def test_canonical_case_insensitive(self) -> None:
        tax = SpeciesTaxonomy()
        assert tax.get_id("neem") == tax.get_id("Neem")
        assert tax.get_id("azadirachta indica") == tax.get_id("Azadirachta indica")

    def test_canonical_underscore_variants(self) -> None:
        tax = SpeciesTaxonomy()
        assert tax.get_id("Azadirachta_indica") == tax.get_id("Azadirachta indica")

    def test_register_new_species(self) -> None:
        tax = SpeciesTaxonomy()
        initial = tax.num_species
        sid = tax.get_or_register("Centella asiatica", dataset_source="test")
        assert sid == initial  # Next ID
        assert tax.num_species == initial + 1

    def test_register_duplicate_returns_same_id(self) -> None:
        tax = SpeciesTaxonomy()
        sid1 = tax.get_or_register("Centella asiatica")
        sid2 = tax.get_or_register("Centella asiatica")
        assert sid1 == sid2
        # Should not increment
        assert tax.get_or_register("centella asiatica") == sid1

    def test_register_canonical_overlap(self) -> None:
        """Registering a canonical species from a dataset should return existing ID."""
        tax = SpeciesTaxonomy()
        neem_id = tax.get_id("Neem")
        assert tax.get_or_register("Neem", dataset_source="herbify") == neem_id
        assert tax.get_or_register("Azadirachta indica", dataset_source="medleafx") == neem_id

    def test_get_name(self) -> None:
        tax = SpeciesTaxonomy()
        name = tax.get_name(0)
        assert name == "Azadirachta indica"

    def test_get_name_unknown(self) -> None:
        tax = SpeciesTaxonomy()
        assert "Unknown" in tax.get_name(9999)

    def test_get_id_not_found(self) -> None:
        tax = SpeciesTaxonomy()
        assert tax.get_id("Nonexistent plantica") is None

    def test_summary(self) -> None:
        tax = SpeciesTaxonomy()
        s = tax.summary()
        assert "10 species registered" in s
        assert "Azadirachta indica" in s


# ─── Pathology Mapping Tests ────────────────────────────────────────────────


class TestPathologyMapping:
    """Tests for raw pathology label -> unified ID mapping."""

    def test_none_maps_to_healthy(self) -> None:
        assert map_pathology_label(None) == 0

    def test_known_labels(self) -> None:
        assert map_pathology_label("Healthy") == 0
        assert map_pathology_label("Bacterial Spot") == 1
        assert map_pathology_label("Shot Hole") == 2
        assert map_pathology_label("Powdery Mildew") == 3
        assert map_pathology_label("Yellow Leaf Disease") == 4
        assert map_pathology_label("Nitrogen Deficiency") == 5
        assert map_pathology_label("Potassium Deficiency") == 6
        assert map_pathology_label("Unhealthy") == 7

    def test_case_variants(self) -> None:
        assert map_pathology_label("healthy") == 0
        assert map_pathology_label("bacterial_spot") == 1
        assert map_pathology_label("shot_hole") == 2
        assert map_pathology_label("powdery_mildew") == 3
        assert map_pathology_label("unhealthy") == 7
        assert map_pathology_label("diseased") == 7

    def test_earlynsd_labels(self) -> None:
        assert map_pathology_label("N_deficiency") == 5
        assert map_pathology_label("K_deficiency") == 6
        assert map_pathology_label("N-deficiency") == 5
        assert map_pathology_label("K-deficiency") == 6

    def test_unknown_label_defaults_healthy(self) -> None:
        assert map_pathology_label("some_random_label") == 0

    def test_pathology_classes_count(self) -> None:
        assert len(PATHOLOGY_CLASSES) == 8
        assert PATHOLOGY_CLASSES[0] == "Healthy"

    def test_pathology_to_id_consistency(self) -> None:
        for i, name in enumerate(PATHOLOGY_CLASSES):
            assert PATHOLOGY_TO_ID[name] == i


# ─── Per-Dataset Loader Tests ───────────────────────────────────────────────


class TestHerbifyLoader:
    """Test load_herbify with synthetic directory structure."""

    def test_basic_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "herbify"
        for species in ["Neem", "Tulsi", "Turmeric"]:
            for i in range(3):
                _make_image(root / species / f"img_{i}.jpg")

        samples = load_herbify(root)
        assert len(samples) == 9
        species_names = {s[1] for s in samples}
        assert species_names == {"Neem", "Tulsi", "Turmeric"}
        # Herbify has no pathology labels
        assert all(s[2] is None for s in samples)

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "herbify"
        _make_image(root / "Neem" / "img.jpg")
        _make_image(root / ".DS_Store" / "junk.jpg")
        samples = load_herbify(root)
        assert len(samples) == 1

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        samples = load_herbify(tmp_path / "does_not_exist")
        assert samples == []

    def test_multiple_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "herbify"
        _make_image(root / "Species1" / "a.jpg")
        _make_image(root / "Species1" / "b.png")
        _make_image(root / "Species1" / "c.jpeg")
        samples = load_herbify(root)
        assert len(samples) == 3

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        root = tmp_path / "herbify"
        _make_image(root / "Species1" / "img.jpg")
        (root / "Species1" / "readme.txt").write_text("hello")
        samples = load_herbify(root)
        assert len(samples) == 1


class TestAssamLoader:
    """Test load_assam — identical layout to Herbify."""

    def test_basic_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "assam"
        for species in ["Adhatoda_vasica", "Centella_asiatica"]:
            for i in range(4):
                _make_image(root / species / f"img_{i}.jpg")
        samples = load_assam(root)
        assert len(samples) == 8
        assert all(s[2] is None for s in samples)


class TestMedLeafXLoader:
    """Test load_medleafx — hierarchical species/condition layout."""

    def test_hierarchical_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "medleafx"
        for species in ["Neem", "Moringa"]:
            for condition in ["Healthy", "Bacterial Spot", "Shot Hole"]:
                for i in range(2):
                    _make_image(root / species / condition / f"img_{i}.jpg")
        samples = load_medleafx(root)
        assert len(samples) == 12
        # All should have pathology labels
        assert all(s[2] is not None for s in samples)
        conditions = {s[2] for s in samples}
        assert "Healthy" in conditions
        assert "Bacterial Spot" in conditions

    def test_flat_layout_with_separator(self, tmp_path: Path) -> None:
        root = tmp_path / "medleafx"
        _make_image(root / "Neem___Bacterial_Spot" / "img.jpg")
        _make_image(root / "Neem___Healthy" / "img.jpg")
        samples = load_medleafx(root)
        assert len(samples) == 2
        species = {s[1] for s in samples}
        assert "Neem" in species

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        assert load_medleafx(tmp_path / "nope") == []


class TestCIMPDLoader:
    """Test load_cimpd — folder-per-species layout (23 species, no pathology labels)."""

    def test_basic_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "cimpd"
        for species in ["Tulsi", "Aloe_Vera", "Neem"]:
            for i in range(4):
                _make_image(root / species / f"img_{i}.jpg")
        samples = load_cimpd(root)
        assert len(samples) == 12
        assert all(s[2] is None for s in samples)
        species = {s[1] for s in samples}
        assert species == {"Tulsi", "Aloe_Vera", "Neem"}

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        assert load_cimpd(tmp_path / "nope") == []

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "cimpd"
        _make_image(root / ".hidden" / "img.jpg")
        _make_image(root / "Tulsi" / "img.jpg")
        samples = load_cimpd(root)
        assert len(samples) == 1


class TestSIMPLoader:
    """Test load_simp — folder-per-species layout."""

    def test_basic_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "simp"
        for species in ["Betel", "Curry_Leaf", "Tamarind"]:
            for i in range(2):
                _make_image(root / species / f"img_{i}.jpg")
        samples = load_simp(root)
        assert len(samples) == 6
        assert all(s[2] is None for s in samples)


class TestEarlyNSDLoader:
    """Test load_earlynsd — crop/condition hierarchical layout."""

    def test_hierarchical_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "earlynsd"
        for crop in ["ash_gourd", "bitter_gourd"]:
            for cond in ["Healthy", "N_deficiency", "K_deficiency"]:
                for i in range(2):
                    _make_image(root / crop / cond / f"img_{i}.jpg")
        samples = load_earlynsd(root)
        assert len(samples) == 12

    def test_split_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "earlynsd"
        for split in ["train", "val", "test"]:
            _make_image(root / split / "ash_gourd_healthy" / "img.jpg")
            _make_image(root / split / "ash_gourd_n_deficiency" / "img.jpg")
        samples = load_earlynsd(root)
        assert len(samples) == 6

    def test_flat_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "earlynsd"
        _make_image(root / "bitter_gourd_k_deficiency" / "img.jpg")
        _make_image(root / "snake_gourd_healthy" / "img.jpg")
        samples = load_earlynsd(root)
        assert len(samples) == 2

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        assert load_earlynsd(tmp_path / "nope") == []


class TestParseEarlyNSDClass:
    """Test _parse_earlynsd_class name parsing."""

    def test_nitrogen_deficiency(self) -> None:
        from phytoveda.data.datasets import _parse_earlynsd_class

        species, cond = _parse_earlynsd_class("ash_gourd_n_deficiency")
        assert cond == "Nitrogen Deficiency"

    def test_potassium_deficiency(self) -> None:
        from phytoveda.data.datasets import _parse_earlynsd_class

        species, cond = _parse_earlynsd_class("bitter_gourd_k_deficiency")
        assert cond == "Potassium Deficiency"

    def test_healthy(self) -> None:
        from phytoveda.data.datasets import _parse_earlynsd_class

        species, cond = _parse_earlynsd_class("snake_gourd_healthy")
        assert cond == "Healthy"
        assert species == "Snake Gourd"

    def test_unknown_defaults_healthy(self) -> None:
        from phytoveda.data.datasets import _parse_earlynsd_class

        _, cond = _parse_earlynsd_class("unknown_class")
        assert cond == "Healthy"


# ─── FederatedBotanicalDataset Tests ────────────────────────────────────────


class TestFederatedBotanicalDataset:
    """Tests for the unified FederatedBotanicalDataset."""

    def _make_samples(self, tmp_path: Path, n: int = 5) -> list[tuple[Path, int, int]]:
        """Create n synthetic samples with real image files."""
        samples = []
        for i in range(n):
            img_path = tmp_path / f"img_{i}.jpg"
            _make_image(img_path, size=(64, 64))
            samples.append((img_path, i % 3, i % 2))
        return samples

    def test_len(self, tmp_path: Path) -> None:
        samples = self._make_samples(tmp_path, 10)
        ds = FederatedBotanicalDataset(samples, image_size=64)
        assert len(ds) == 10

    def test_getitem_returns_correct_types(self, tmp_path: Path) -> None:
        samples = self._make_samples(tmp_path, 3)
        ds = FederatedBotanicalDataset(samples, image_size=64)
        img, sid, pid = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(sid, int)
        assert isinstance(pid, int)

    def test_getitem_no_transform(self, tmp_path: Path) -> None:
        """Without transform, should resize to image_size and return float tensor."""
        samples = self._make_samples(tmp_path, 1)
        ds = FederatedBotanicalDataset(samples, transform=None, image_size=32)
        img, _, _ = ds[0]
        assert img.shape == (3, 32, 32)
        assert img.dtype == torch.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_getitem_with_val_transform(self, tmp_path: Path) -> None:
        """With val transform, should normalize with ImageNet stats."""
        samples = self._make_samples(tmp_path, 1)
        transform = get_val_transforms(image_size=64)
        ds = FederatedBotanicalDataset(samples, transform=transform, image_size=64)
        img, _, _ = ds[0]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32

    def test_getitem_with_train_transform(self, tmp_path: Path) -> None:
        """With train transform, should apply augmentation."""
        samples = self._make_samples(tmp_path, 1)
        transform = get_train_transforms(image_size=64)
        ds = FederatedBotanicalDataset(samples, transform=transform, image_size=64)
        img, _, _ = ds[0]
        assert img.shape == (3, 64, 64)

    def test_corrupt_image_returns_black(self, tmp_path: Path) -> None:
        """Corrupt image should return a zero tensor, not raise."""
        corrupt_path = tmp_path / "corrupt.jpg"
        _make_corrupt_file(corrupt_path)
        samples = [(corrupt_path, 0, 0)]
        ds = FederatedBotanicalDataset(samples, transform=None, image_size=32)
        img, sid, pid = ds[0]
        assert img.shape == (3, 32, 32)
        # Black image = all zeros
        assert img.sum().item() == 0.0

    def test_species_labels_property(self, tmp_path: Path) -> None:
        samples = self._make_samples(tmp_path, 6)
        ds = FederatedBotanicalDataset(samples)
        labels = ds.species_labels
        assert len(labels) == 6
        assert labels == [s[1] for s in samples]

    def test_pathology_labels_property(self, tmp_path: Path) -> None:
        samples = self._make_samples(tmp_path, 6)
        ds = FederatedBotanicalDataset(samples)
        labels = ds.pathology_labels
        assert len(labels) == 6
        assert labels == [s[2] for s in samples]

    def test_empty_dataset(self) -> None:
        ds = FederatedBotanicalDataset([])
        assert len(ds) == 0


# ─── build_datasets Tests ──────────────────────────────────────────────────


class TestBuildDatasets:
    """Tests for the dataset builder and stratified splitting."""

    def _create_mock_datasets(self, tmp_path: Path) -> Path:
        """Create minimal mock data for 2 datasets."""
        data_root = tmp_path / "data"

        # Herbify: 3 species, 10 images each
        for species in ["Neem", "Tulsi", "Turmeric"]:
            for i in range(10):
                _make_image(data_root / "herbify" / species / f"img_{i}.jpg")

        # MedLeafX: 1 species, 2 conditions, 10 images each
        for condition in ["Healthy", "Bacterial Spot"]:
            for i in range(10):
                _make_image(data_root / "medleafx" / "Neem" / condition / f"img_{i}.jpg")

        return data_root

    def test_build_datasets_basic(self, tmp_path: Path) -> None:
        data_root = self._create_mock_datasets(tmp_path)
        train_ds, val_ds, test_ds, taxonomy = build_datasets(
            data_root,
            datasets=["herbify", "medleafx"],
            image_size=64,
        )
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 50  # 30 herbify + 20 medleafx
        assert taxonomy.num_species >= 3  # Neem, Tulsi, Turmeric (+ canonical)

    def test_split_ratios(self, tmp_path: Path) -> None:
        data_root = self._create_mock_datasets(tmp_path)
        train_ds, val_ds, test_ds, _ = build_datasets(
            data_root,
            datasets=["herbify", "medleafx"],
            image_size=64,
        )
        total = len(train_ds) + len(val_ds) + len(test_ds)
        # Train should be ~70%, val ~15%, test ~15%
        assert len(train_ds) / total == pytest.approx(0.70, abs=0.05)
        assert len(val_ds) / total == pytest.approx(0.15, abs=0.05)
        assert len(test_ds) / total == pytest.approx(0.15, abs=0.05)

    def test_taxonomy_unifies_neem(self, tmp_path: Path) -> None:
        """Neem from herbify and medleafx should get the same species ID."""
        data_root = self._create_mock_datasets(tmp_path)
        _, _, _, taxonomy = build_datasets(
            data_root,
            datasets=["herbify", "medleafx"],
            image_size=64,
        )
        neem_id = taxonomy.get_id("Neem")
        assert neem_id is not None
        # Should be the canonical ID (0)
        assert neem_id == 0

    def test_empty_datasets(self, tmp_path: Path) -> None:
        """Should handle no data gracefully."""
        data_root = tmp_path / "empty"
        data_root.mkdir()
        train_ds, val_ds, test_ds, taxonomy = build_datasets(
            data_root,
            datasets=["herbify"],
            image_size=64,
        )
        assert len(train_ds) == 0
        assert len(val_ds) == 0
        assert len(test_ds) == 0

    def test_unknown_dataset_skipped(self, tmp_path: Path) -> None:
        """Unknown dataset keys should be skipped with a warning."""
        data_root = self._create_mock_datasets(tmp_path)
        train_ds, _, _, _ = build_datasets(
            data_root,
            datasets=["herbify", "nonexistent_dataset"],
            image_size=64,
        )
        assert len(train_ds) > 0

    def test_reproducible_splits(self, tmp_path: Path) -> None:
        """Same seed should produce identical splits."""
        data_root = self._create_mock_datasets(tmp_path)
        train1, val1, test1, _ = build_datasets(
            data_root, datasets=["herbify"], image_size=64, seed=123,
        )
        train2, val2, test2, _ = build_datasets(
            data_root, datasets=["herbify"], image_size=64, seed=123,
        )
        # Same samples in same order
        assert [s[0] for s in train1.samples] == [s[0] for s in train2.samples]
        assert [s[0] for s in val1.samples] == [s[0] for s in val2.samples]

    def test_different_seed_different_splits(self, tmp_path: Path) -> None:
        data_root = self._create_mock_datasets(tmp_path)
        train1, _, _, _ = build_datasets(
            data_root, datasets=["herbify"], image_size=64, seed=1,
        )
        train2, _, _, _ = build_datasets(
            data_root, datasets=["herbify"], image_size=64, seed=999,
        )
        # Very unlikely to be the same
        paths1 = [s[0] for s in train1.samples]
        paths2 = [s[0] for s in train2.samples]
        assert paths1 != paths2


# ─── Weighted Sampler Tests ─────────────────────────────────────────────────


class TestBuildWeightedSampler:
    """Tests for class-balanced sampling."""

    def test_sampler_length(self, tmp_path: Path) -> None:
        samples = []
        for i in range(20):
            img = tmp_path / f"img_{i}.jpg"
            _make_image(img)
            samples.append((img, i % 4, 0))
        ds = FederatedBotanicalDataset(samples)
        sampler = build_weighted_sampler(ds)
        assert sampler.num_samples == 20

    def test_rare_class_gets_higher_weight(self, tmp_path: Path) -> None:
        """A species with fewer samples should have higher sampling weight."""
        samples = []
        # 18 images of species 0, 2 images of species 1
        for i in range(18):
            img = tmp_path / f"common_{i}.jpg"
            _make_image(img)
            samples.append((img, 0, 0))
        for i in range(2):
            img = tmp_path / f"rare_{i}.jpg"
            _make_image(img)
            samples.append((img, 1, 0))

        ds = FederatedBotanicalDataset(samples)
        sampler = build_weighted_sampler(ds)
        weights = list(sampler.weights)
        # Weight for rare (species 1) should be higher than common (species 0)
        common_weight = weights[0]
        rare_weight = weights[18]
        assert rare_weight > common_weight


# ─── Augmentation Tests ─────────────────────────────────────────────────────


class TestAugmentation:
    """Tests for training and validation transform pipelines."""

    def test_train_transforms_output_shape(self) -> None:
        transform = get_train_transforms(image_size=128)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert result.shape == (3, 128, 128)
        assert result.dtype == torch.float32

    def test_val_transforms_output_shape(self) -> None:
        transform = get_val_transforms(image_size=128)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert result.shape == (3, 128, 128)
        assert result.dtype == torch.float32

    def test_val_transforms_deterministic(self) -> None:
        """Val transforms should produce identical output for same input."""
        transform = get_val_transforms(image_size=64)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        r1 = transform(image=img)["image"]
        r2 = transform(image=img)["image"]
        assert torch.equal(r1, r2)

    def test_train_transforms_normalized(self) -> None:
        """Training output should be roughly centered around 0 (ImageNet normalization)."""
        transform = get_train_transforms(image_size=64)
        # A gray image (128, 128, 128) should normalize to near 0
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = transform(image=img)["image"]
        # After ImageNet normalization, 128/255 ≈ 0.502 → (0.502-0.485)/0.229 ≈ 0.07 for R
        assert result.mean().abs() < 2.0  # Reasonable range

    def test_custom_image_size(self) -> None:
        for size in [64, 256, 512]:
            transform = get_val_transforms(image_size=size)
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = transform(image=img)["image"]
            assert result.shape == (3, size, size)


# ─── Preprocessing Tests ───────────────────────────────────────────────────


class TestPreprocessing:
    """Tests for image preprocessing utilities."""

    def test_preprocess_image(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.jpg"
        _make_image(img_path, size=(100, 80))
        result = preprocess_image(img_path)
        assert result is not None
        assert result.size == (512, 512)
        assert result.mode == "RGB"

    def test_preprocess_corrupt_image(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        _make_corrupt_file(path)
        assert preprocess_image(path) is None

    def test_preprocess_nonexistent(self, tmp_path: Path) -> None:
        assert preprocess_image(tmp_path / "nope.jpg") is None

    def test_validate_image_valid(self, tmp_path: Path) -> None:
        img_path = tmp_path / "valid.jpg"
        _make_image(img_path)
        assert validate_image(img_path) is True

    def test_validate_image_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        _make_corrupt_file(path)
        assert validate_image(path) is False

    def test_validate_image_nonexistent(self, tmp_path: Path) -> None:
        assert validate_image(tmp_path / "nope.jpg") is False


# ─── Download / Validation Tests ───────────────────────────────────────────


class TestDownloadUtilities:
    """Tests for download-related helpers (no actual downloads)."""

    def test_count_images_empty_dir(self, tmp_path: Path) -> None:
        assert count_images(tmp_path) == 0

    def test_count_images_nonexistent(self, tmp_path: Path) -> None:
        assert count_images(tmp_path / "nope") == 0

    def test_count_images_mixed(self, tmp_path: Path) -> None:
        _make_image(tmp_path / "a.jpg")
        _make_image(tmp_path / "b.png")
        _make_image(tmp_path / "sub" / "c.jpeg")
        (tmp_path / "readme.txt").write_text("not an image")
        assert count_images(tmp_path) == 3

    def test_validate_dataset_missing(self, tmp_path: Path) -> None:
        is_valid, msg = validate_dataset(tmp_path, "herbify")
        assert not is_valid
        assert "not found" in msg

    def test_validate_dataset_insufficient(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "herbify"
        # Only 10 images when 6104 expected
        for i in range(10):
            _make_image(ds_dir / f"img_{i}.jpg")
        is_valid, msg = validate_dataset(tmp_path, "herbify")
        assert not is_valid
        assert "found 10" in msg

    def test_validate_dataset_sufficient(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "earlynsd"
        # EarlyNSD expects 2700, 90% = 2430
        for i in range(2500):
            _make_image(ds_dir / f"img_{i}.jpg")
        is_valid, msg = validate_dataset(tmp_path, "earlynsd")
        assert is_valid
        assert "validated" in msg

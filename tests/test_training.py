"""Tests for the training pipeline.

Uses vit_tiny_patch16_224 (untrained) as a lightweight backbone for fast testing.
All tests use synthetic data — no real datasets or GPU required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from phytoveda.data.augmentation import get_val_transforms
from phytoveda.data.datasets import FederatedBotanicalDataset
from phytoveda.models.cmtl import HierViTCMTL
from phytoveda.training.evaluation import DetailedMetrics, evaluate, evaluate_detailed
from phytoveda.training.trainer import CheckpointManager, TrainConfig, Trainer, load_config

# ─── Fixtures ───────────────────────────────────────────────────────────────

TINY_BACKBONE = "vit_tiny_patch16_224"
TINY_IMG_SIZE = 224
NUM_SPECIES = 5
NUM_PATHOLOGIES = 3


def _make_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(path, format="JPEG")


@pytest.fixture()
def tiny_model() -> HierViTCMTL:
    """Small ViT model for testing — no pretrained weights."""
    return HierViTCMTL(
        num_species=NUM_SPECIES,
        num_pathologies=NUM_PATHOLOGIES,
        backbone_name=TINY_BACKBONE,
        pretrained=False,
        image_size=TINY_IMG_SIZE,
    )


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> FederatedBotanicalDataset:
    """20-sample synthetic dataset with val transforms."""
    samples = []
    for i in range(20):
        img_path = tmp_path / f"img_{i}.jpg"
        _make_image(img_path, size=(TINY_IMG_SIZE, TINY_IMG_SIZE))
        samples.append((img_path, i % NUM_SPECIES, i % NUM_PATHOLOGIES))
    transform = get_val_transforms(image_size=TINY_IMG_SIZE)
    return FederatedBotanicalDataset(samples, transform=transform, image_size=TINY_IMG_SIZE)


@pytest.fixture()
def synthetic_dataloader(synthetic_dataset: FederatedBotanicalDataset) -> DataLoader:
    return DataLoader(synthetic_dataset, batch_size=4, shuffle=False, num_workers=0)


# ─── TrainConfig Tests ──────────────────────────────────────────────────────


class TestTrainConfig:
    def test_defaults(self) -> None:
        config = TrainConfig()
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.warmup_epochs == 5
        assert config.save_top_k == 3
        assert config.mixed_precision is True

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
model:
  backbone: "vit_tiny_patch16_224"
  pretrained: false
  image_size: 224
  dropout: 0.5

species:
  num_classes: 50

pathology:
  num_classes: 4

training:
  epochs: 10
  batch_size: 8
  learning_rate: 0.001
  warmup_epochs: 2

loss:
  focal_gamma: 3.0
  reg_lambda: 0.05

data:
  num_workers: 2
  datasets:
    - herbify
    - assam

checkpoint:
  dir: "ckpts"
  save_top_k: 5
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = TrainConfig.from_yaml(config_path)
        assert config.backbone == "vit_tiny_patch16_224"
        assert config.pretrained is False
        assert config.image_size == 224
        assert config.dropout == 0.5
        assert config.num_species == 50
        assert config.num_pathologies == 4
        assert config.epochs == 10
        assert config.batch_size == 8
        assert config.learning_rate == 0.001
        assert config.warmup_epochs == 2
        assert config.focal_gamma == 3.0
        assert config.reg_lambda == 0.05
        assert config.num_workers == 2
        assert config.datasets == ["herbify", "assam"]
        assert config.checkpoint_dir == "ckpts"
        assert config.save_top_k == 5

    def test_from_yaml_partial(self, tmp_path: Path) -> None:
        """Missing keys should keep defaults."""
        config_path = tmp_path / "partial.yaml"
        config_path.write_text("training:\n  epochs: 5\n")
        config = TrainConfig.from_yaml(config_path)
        assert config.epochs == 5
        assert config.learning_rate == 1e-4  # Default

    def test_load_config_with_overrides(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("training:\n  epochs: 50\n")
        config = load_config(config_path, overrides={"epochs": 10, "batch_size": 32})
        assert config.epochs == 10
        assert config.batch_size == 32

    def test_load_config_none_overrides_ignored(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("training:\n  epochs: 50\n")
        config = load_config(config_path, overrides={"epochs": None})
        assert config.epochs == 50

    def test_real_config_loads(self) -> None:
        """The actual project config file should load without errors."""
        config_path = Path("configs/hiervit_cmtl.yaml")
        if config_path.exists():
            config = TrainConfig.from_yaml(config_path)
            assert config.epochs > 0
            assert config.num_species > 0


# ─── CheckpointManager Tests ───────────────────────────────────────────────


class TestCheckpointManager:
    def test_save_first_checkpoint(self, tmp_path: Path, tiny_model: HierViTCMTL) -> None:
        mgr = CheckpointManager(tmp_path / "ckpts", save_top_k=3)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        saved = mgr.save_if_improved(0.5, epoch=1, model=tiny_model, optimizer=optimizer, scheduler=optimizer)
        assert saved
        assert mgr.best_f1 == 0.5
        assert (tmp_path / "ckpts" / "best_model.pt").exists()

    def test_save_top_k_eviction(self, tmp_path: Path, tiny_model: HierViTCMTL) -> None:
        mgr = CheckpointManager(tmp_path / "ckpts", save_top_k=2)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        mgr.save_if_improved(0.3, epoch=1, model=tiny_model, optimizer=optimizer, scheduler=optimizer)
        mgr.save_if_improved(0.5, epoch=2, model=tiny_model, optimizer=optimizer, scheduler=optimizer)
        mgr.save_if_improved(0.7, epoch=3, model=tiny_model, optimizer=optimizer, scheduler=optimizer)

        # Should have evicted worst (0.3)
        assert mgr.best_f1 == 0.7
        ckpts = list((tmp_path / "ckpts").glob("checkpoint_epoch*.pt"))
        assert len(ckpts) == 2  # top-2 kept

    def test_no_save_if_not_improved(self, tmp_path: Path, tiny_model: HierViTCMTL) -> None:
        mgr = CheckpointManager(tmp_path / "ckpts", save_top_k=1)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        mgr.save_if_improved(0.8, epoch=1, model=tiny_model, optimizer=optimizer, scheduler=optimizer)
        saved = mgr.save_if_improved(0.3, epoch=2, model=tiny_model, optimizer=optimizer, scheduler=optimizer)
        assert not saved
        assert mgr.best_f1 == 0.8

    def test_best_f1_empty(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpts")
        assert mgr.best_f1 == 0.0

    def test_checkpoint_contents(self, tmp_path: Path, tiny_model: HierViTCMTL) -> None:
        mgr = CheckpointManager(tmp_path / "ckpts", save_top_k=1)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        mgr.save_if_improved(
            0.9, epoch=5, model=tiny_model, optimizer=optimizer, scheduler=optimizer,
            extra={"val_metrics": {"species_f1": 0.9}},
        )
        ckpt = torch.load(tmp_path / "ckpts" / "best_model.pt", weights_only=False)
        assert ckpt["epoch"] == 5
        assert ckpt["f1_score"] == 0.9
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["val_metrics"]["species_f1"] == 0.9


# ─── Evaluation Tests ──────────────────────────────────────────────────────


class TestEvaluation:
    def test_evaluate_returns_all_keys(
        self, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader
    ) -> None:
        device = torch.device("cpu")
        tiny_model.to(device)
        metrics = evaluate(tiny_model, synthetic_dataloader, device)

        assert "species_f1" in metrics
        assert "pathology_f1" in metrics
        assert "species_acc" in metrics
        assert "pathology_acc" in metrics
        assert "species_top5_acc" in metrics

    def test_evaluate_ranges(
        self, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader
    ) -> None:
        device = torch.device("cpu")
        tiny_model.to(device)
        metrics = evaluate(tiny_model, synthetic_dataloader, device)

        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_top5_accuracy_gte_top1(
        self, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader
    ) -> None:
        device = torch.device("cpu")
        tiny_model.to(device)
        metrics = evaluate(tiny_model, synthetic_dataloader, device)
        assert metrics["species_top5_acc"] >= metrics["species_acc"]

    def test_evaluate_detailed_returns_reports(
        self, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader
    ) -> None:
        device = torch.device("cpu")
        tiny_model.to(device)
        detailed = evaluate_detailed(tiny_model, synthetic_dataloader, device)

        assert isinstance(detailed, DetailedMetrics)
        assert isinstance(detailed.species_report, str)
        assert len(detailed.species_report) > 0
        assert isinstance(detailed.pathology_report, str)
        assert len(detailed.pathology_report) > 0
        assert detailed.species_confusion.shape[0] > 0
        assert detailed.pathology_confusion.shape[0] > 0

    def test_evaluate_detailed_with_names(
        self, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader
    ) -> None:
        device = torch.device("cpu")
        tiny_model.to(device)
        species_names = [f"Species_{i}" for i in range(NUM_SPECIES)]
        pathology_names = [f"Pathology_{i}" for i in range(NUM_PATHOLOGIES)]

        detailed = evaluate_detailed(
            tiny_model, synthetic_dataloader, device,
            species_names=species_names,
            pathology_names=pathology_names,
        )
        assert "Species_0" in detailed.species_report
        assert "Pathology_0" in detailed.pathology_report


# ─── Trainer Integration Tests ──────────────────────────────────────────────


class TestTrainer:
    def _make_config(self, tmp_path: Path) -> TrainConfig:
        return TrainConfig(
            backbone=TINY_BACKBONE,
            pretrained=False,
            image_size=TINY_IMG_SIZE,
            num_species=NUM_SPECIES,
            num_pathologies=NUM_PATHOLOGIES,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            warmup_epochs=1,
            mixed_precision=False,
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_top_k=2,
        )

    def test_trainer_init(
        self,
        tmp_path: Path,
        tiny_model: HierViTCMTL,
        synthetic_dataloader: DataLoader,
    ) -> None:
        config = self._make_config(tmp_path)
        trainer = Trainer(
            tiny_model, synthetic_dataloader, synthetic_dataloader, config,
            device=torch.device("cpu"),
        )
        assert trainer.device == torch.device("cpu")
        assert trainer.amp_enabled is False

    def test_train_epoch(
        self,
        tmp_path: Path,
        tiny_model: HierViTCMTL,
        synthetic_dataloader: DataLoader,
    ) -> None:
        config = self._make_config(tmp_path)
        trainer = Trainer(
            tiny_model, synthetic_dataloader, synthetic_dataloader, config,
            device=torch.device("cpu"),
        )
        losses = trainer.train_epoch()
        assert "total" in losses
        assert "species" in losses
        assert "disease" in losses
        assert all(v >= 0 for v in losses.values())

    def test_full_train_loop(
        self,
        tmp_path: Path,
        tiny_model: HierViTCMTL,
        synthetic_dataloader: DataLoader,
    ) -> None:
        config = self._make_config(tmp_path)
        trainer = Trainer(
            tiny_model, synthetic_dataloader, synthetic_dataloader, config,
            device=torch.device("cpu"),
        )
        val_metrics = trainer.train()

        assert "species_f1" in val_metrics
        assert "pathology_f1" in val_metrics
        # Should have created at least one checkpoint
        assert (tmp_path / "ckpts" / "best_model.pt").exists()

    def test_warmup_scheduler(self, tmp_path: Path, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader) -> None:
        """LR should increase during warmup then decrease."""
        config = self._make_config(tmp_path)
        config.epochs = 4
        config.warmup_epochs = 2
        trainer = Trainer(
            tiny_model, synthetic_dataloader, synthetic_dataloader, config,
            device=torch.device("cpu"),
        )

        lrs = []
        for _ in range(4):
            trainer.train_epoch()
            lrs.append(trainer.optimizer.param_groups[0]["lr"])

        # During warmup (epochs 1-2) LR should increase
        assert lrs[1] > lrs[0], f"LR should increase during warmup: {lrs}"

    def test_no_warmup(self, tmp_path: Path, tiny_model: HierViTCMTL, synthetic_dataloader: DataLoader) -> None:
        config = self._make_config(tmp_path)
        config.warmup_epochs = 0
        config.epochs = 2
        trainer = Trainer(
            tiny_model, synthetic_dataloader, synthetic_dataloader, config,
            device=torch.device("cpu"),
        )
        # Should not crash
        losses = trainer.train_epoch()
        assert losses["total"] >= 0

    def test_checkpoint_resume(
        self,
        tmp_path: Path,
        tiny_model: HierViTCMTL,
        synthetic_dataloader: DataLoader,
    ) -> None:
        """Save a checkpoint and verify it can be loaded."""
        config = self._make_config(tmp_path)
        trainer = Trainer(
            tiny_model, synthetic_dataloader, synthetic_dataloader, config,
            device=torch.device("cpu"),
        )
        trainer.train()

        # Load checkpoint into a fresh model
        ckpt_path = tmp_path / "ckpts" / "best_model.pt"
        assert ckpt_path.exists()

        fresh_model = HierViTCMTL(
            num_species=NUM_SPECIES,
            num_pathologies=NUM_PATHOLOGIES,
            backbone_name=TINY_BACKBONE,
            pretrained=False,
            image_size=TINY_IMG_SIZE,
        )
        ckpt = torch.load(ckpt_path, weights_only=False)
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        # Fresh model should produce same outputs as trained model
        tiny_model.eval()
        fresh_model.eval()
        test_input = torch.randn(1, 3, TINY_IMG_SIZE, TINY_IMG_SIZE)
        orig_out = tiny_model(test_input)
        loaded_out = fresh_model(test_input)
        assert torch.allclose(orig_out[0], loaded_out[0], atol=1e-5)
        assert torch.allclose(orig_out[1], loaded_out[1], atol=1e-5)

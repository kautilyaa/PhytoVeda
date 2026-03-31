"""Tests for Colab training optimizations.

Tests gradient accumulation, crash checkpointing, GPU monitor,
batch size finder, torch.compile wrapper, and dataset caching.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from phytoveda.colab.data_cache import DatasetCache, compute_split_hash
from phytoveda.colab.training import (
    ColabTrainer,
    CrashCheckpointer,
    GPUMonitor,
    GradAccumConfig,
    compile_model,
    find_max_batch_size,
)


# ─── GradAccumConfig ─────────────────────────────────────────────────────


class TestGradAccumConfig:
    def test_no_accumulation(self) -> None:
        config = GradAccumConfig(accumulation_steps=1)
        assert not config.is_accumulating
        assert config.should_step(0)
        assert config.should_step(1)

    def test_accumulate_4(self) -> None:
        config = GradAccumConfig(accumulation_steps=4)
        assert config.is_accumulating
        assert not config.should_step(0)  # batch 0
        assert not config.should_step(1)  # batch 1
        assert not config.should_step(2)  # batch 2
        assert config.should_step(3)      # batch 3 → step

    def test_accumulate_8(self) -> None:
        config = GradAccumConfig(accumulation_steps=8)
        steps = [i for i in range(32) if config.should_step(i)]
        assert steps == [7, 15, 23, 31]

    def test_scale_loss_no_accum(self) -> None:
        config = GradAccumConfig(accumulation_steps=1)
        loss = torch.tensor(4.0)
        scaled = config.scale_loss(loss)
        assert scaled.item() == 4.0

    def test_scale_loss_with_accum(self) -> None:
        config = GradAccumConfig(accumulation_steps=4)
        loss = torch.tensor(4.0)
        scaled = config.scale_loss(loss)
        assert scaled.item() == pytest.approx(1.0)

    def test_scale_loss_8(self) -> None:
        config = GradAccumConfig(accumulation_steps=8)
        loss = torch.tensor(8.0)
        scaled = config.scale_loss(loss)
        assert scaled.item() == pytest.approx(1.0)


# ─── CrashCheckpointer ──────────────────────────────────────────────────


class TestCrashCheckpointer:
    def test_crash_checkpoint_path(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path, interval_minutes=1)
        assert cc.crash_checkpoint_path == tmp_path / "crash_recovery.pt"

    def test_no_recovery_initially(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path)
        assert not cc.has_recovery()

    def test_maybe_save_respects_interval(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path, interval_minutes=60)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # First call: should NOT save (just started)
        saved = cc.maybe_save(model, opt, None, epoch=0, batch_idx=0)
        assert not saved

    def test_maybe_save_after_interval(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path, interval_minutes=0.0001)  # ~6ms
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Wait a tiny bit then save
        time.sleep(0.01)
        saved = cc.maybe_save(model, opt, None, epoch=1, batch_idx=10)
        assert saved
        assert cc.has_recovery()
        assert cc.save_count == 1

    def test_load_recovery(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path, interval_minutes=0)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Force save
        cc._last_save_time = 0  # pretend we haven't saved in ages
        cc.maybe_save(model, opt, None, epoch=3, batch_idx=42, metrics={"loss": 0.5})

        # Create fresh model and load
        model2 = nn.Linear(4, 2)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        info = cc.load_recovery(model2, opt2, None, device="cpu")

        assert info["epoch"] == 3
        assert info["batch_idx"] == 42
        assert info["metrics"]["loss"] == 0.5

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_cleanup(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path, interval_minutes=0)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        cc._last_save_time = 0
        cc.maybe_save(model, opt, None, epoch=0, batch_idx=0)
        assert cc.has_recovery()

        cc.cleanup()
        assert not cc.has_recovery()

    def test_save_count_increments(self, tmp_path: Path) -> None:
        cc = CrashCheckpointer(save_dir=tmp_path, interval_minutes=0)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        for i in range(3):
            cc._last_save_time = 0
            cc.maybe_save(model, opt, None, epoch=0, batch_idx=i)

        assert cc.save_count == 3


# ─── GPUMonitor ──────────────────────────────────────────────────────────


class TestGPUMonitor:
    def test_snapshot_no_gpu(self) -> None:
        monitor = GPUMonitor(device="cpu")
        monitor._available = False
        result = monitor.snapshot("test")
        assert result is None

    def test_snapshots_list_empty(self) -> None:
        monitor = GPUMonitor()
        monitor._available = False
        assert monitor.snapshots == []

    def test_peak_allocated_no_gpu(self) -> None:
        monitor = GPUMonitor()
        monitor._available = False
        assert monitor.peak_allocated_mb() == 0.0

    def test_warn_if_low_no_gpu(self) -> None:
        monitor = GPUMonitor()
        monitor._available = False
        assert monitor.warn_if_low() is False

    def test_summary_empty(self) -> None:
        monitor = GPUMonitor()
        monitor._available = False
        assert "No GPU snapshots" in monitor.summary()

    def test_clear_cache_no_crash(self) -> None:
        """clear_cache should not crash even without GPU."""
        monitor = GPUMonitor()
        monitor._available = False
        monitor.clear_cache()  # Should not raise


# ─── compile_model ───────────────────────────────────────────────────────


class TestCompileModel:
    def test_fallback_on_failure(self) -> None:
        model = nn.Linear(4, 2)
        # Patch torch.compile to raise
        with patch("torch.compile", side_effect=RuntimeError("not supported")):
            result = compile_model(model, fallback=True)
        assert result is model  # Should return original

    def test_no_fallback_raises(self) -> None:
        model = nn.Linear(4, 2)
        with patch("torch.compile", side_effect=RuntimeError("not supported")):
            with pytest.raises(RuntimeError):
                compile_model(model, fallback=False)

    def test_compile_succeeds(self) -> None:
        model = nn.Linear(4, 2)
        # torch.compile should work on simple models (PyTorch 2.x)
        result = compile_model(model, mode="default", fallback=True)
        assert result is not None


# ─── find_max_batch_size ─────────────────────────────────────────────────


class TestFindMaxBatchSize:
    def test_returns_min_without_gpu(self) -> None:
        model = nn.Linear(4, 2)
        result = find_max_batch_size(model, sample_input_shape=(4,), device="cpu")
        assert result == 1  # min_batch default

    def test_custom_min_batch(self) -> None:
        model = nn.Linear(4, 2)
        result = find_max_batch_size(
            model, sample_input_shape=(4,), device="cpu", min_batch=4
        )
        assert result == 4


# ─── DatasetCache ────────────────────────────────────────────────────────


class TestDatasetCache:
    def test_init_creates_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache = DatasetCache(cache_dir)
        assert cache_dir.exists()

    # ── Splits ──

    def test_save_and_load_splits(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        train = [0, 1, 2, 3]
        val = [4, 5]
        test = [6, 7]
        cache.save_splits(train, val, test, split_hash="abc123")

        assert cache.has_splits()
        assert cache.has_splits("abc123")
        assert not cache.has_splits("wrong_hash")

        t, v, te = cache.load_splits()
        assert t == train
        assert v == val
        assert te == test

    def test_no_splits_initially(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        assert not cache.has_splits()

    def test_load_splits_not_found(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        with pytest.raises(FileNotFoundError):
            cache.load_splits()

    def test_has_splits_no_hash_check(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        cache.save_splits([0], [1], [2])
        assert cache.has_splits()  # No hash check

    # ── Taxonomy ──

    def test_save_and_load_taxonomy(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        species = {"Neem": 0, "Tulsi": 1}
        pathology = {"Healthy": 0, "Bacterial Spot": 1}

        cache.save_taxonomy(species, pathology)
        assert cache.has_taxonomy()

        tax = cache.load_taxonomy()
        assert tax["num_species"] == 2
        assert tax["num_pathology"] == 2
        assert tax["species"]["Neem"] == 0

    def test_no_taxonomy_initially(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        assert not cache.has_taxonomy()

    def test_load_taxonomy_not_found(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        with pytest.raises(FileNotFoundError):
            cache.load_taxonomy()

    # ── Download Status ──

    def test_mark_downloaded(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        data_dir = tmp_path / "datasets" / "herbify"
        data_dir.mkdir(parents=True)

        cache.mark_downloaded("herbify", 6104, str(data_dir))
        assert cache.is_downloaded("herbify")
        assert cache.is_downloaded("herbify", min_images=6000)
        assert not cache.is_downloaded("herbify", min_images=10000)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        assert not cache.is_downloaded("nonexistent")

    def test_downloaded_but_path_missing(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        cache.mark_downloaded("herbify", 6104, "/nonexistent/path")
        assert not cache.is_downloaded("herbify")  # Path doesn't exist

    def test_download_status(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        d1 = tmp_path / "d1"
        d1.mkdir()
        cache.mark_downloaded("herbify", 6104, str(d1))
        cache.mark_downloaded("assam", 7341, str(d1))

        status = cache.download_status()
        assert "herbify" in status
        assert "assam" in status
        assert status["herbify"]["image_count"] == 6104

    # ── Training History ──

    def test_save_and_load_history(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        history = {
            "train_loss": [0.9, 0.7, 0.5],
            "species_f1": [0.3, 0.5, 0.7],
        }
        cache.save_history(history)
        assert cache.has_history()

        loaded = cache.load_history()
        assert loaded["train_loss"] == [0.9, 0.7, 0.5]

    def test_no_history_initially(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        assert not cache.has_history()

    def test_load_history_not_found(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        with pytest.raises(FileNotFoundError):
            cache.load_history()

    # ── Utilities ──

    def test_clear(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        cache.save_splits([0], [1], [2])
        cache.save_taxonomy({"A": 0}, {"B": 0})
        count = cache.clear()
        assert count >= 2
        assert not cache.has_splits()
        assert not cache.has_taxonomy()

    def test_summary(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path / "cache")
        cache.save_splits([0, 1, 2], [3], [4])
        cache.save_taxonomy({"Neem": 0}, {"Healthy": 0})

        summary = cache.summary()
        assert "Splits" in summary
        assert "Taxonomy" in summary
        assert "3 train" in summary


# ─── compute_split_hash ──────────────────────────────────────────────────


class TestComputeSplitHash:
    def test_deterministic(self) -> None:
        h1 = compute_split_hash(["herbify", "assam"], 0.7, 0.15)
        h2 = compute_split_hash(["herbify", "assam"], 0.7, 0.15)
        assert h1 == h2

    def test_order_independent(self) -> None:
        h1 = compute_split_hash(["herbify", "assam"], 0.7, 0.15)
        h2 = compute_split_hash(["assam", "herbify"], 0.7, 0.15)
        assert h1 == h2

    def test_different_ratios(self) -> None:
        h1 = compute_split_hash(["herbify"], 0.7, 0.15)
        h2 = compute_split_hash(["herbify"], 0.8, 0.10)
        assert h1 != h2

    def test_different_datasets(self) -> None:
        h1 = compute_split_hash(["herbify"], 0.7, 0.15)
        h2 = compute_split_hash(["assam"], 0.7, 0.15)
        assert h1 != h2

    def test_different_seeds(self) -> None:
        h1 = compute_split_hash(["herbify"], 0.7, 0.15, seed=42)
        h2 = compute_split_hash(["herbify"], 0.7, 0.15, seed=123)
        assert h1 != h2

    def test_hash_length(self) -> None:
        h = compute_split_hash(["herbify"], 0.7, 0.15)
        assert len(h) == 16


# ─── ColabTrainer (unit-level) ───────────────────────────────────────────


class TestColabTrainerInit:
    """Test ColabTrainer initialization without full training."""

    def _make_tiny_model(self) -> nn.Module:
        """Build a minimal CMTL-like model for testing."""
        from phytoveda.models.cmtl import HierViTCMTL

        return HierViTCMTL(
            num_species=4,
            num_pathologies=3,
            backbone_name="vit_tiny_patch16_224",
            pretrained=False,
            image_size=32,
            species_hidden_dim=16,
            pathology_hidden_dim=16,
            dropout=0.0,
        )

    def _make_loader(self, n: int = 8, batch_size: int = 4) -> torch.utils.data.DataLoader:
        from torch.utils.data import DataLoader, TensorDataset

        images = torch.randn(n, 3, 32, 32)
        species = torch.randint(0, 4, (n,))
        pathology = torch.randint(0, 3, (n,))
        ds = TensorDataset(images, species, pathology)
        return DataLoader(ds, batch_size=batch_size)

    def test_init(self, tmp_path: Path) -> None:
        model = self._make_tiny_model()
        loader = self._make_loader()
        config = TrainConfig(epochs=1, checkpoint_dir=str(tmp_path))

        trainer = ColabTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
            accumulation_steps=2,
            crash_interval_minutes=60,
        )

        assert trainer.grad_accum.accumulation_steps == 2
        assert trainer.grad_accum.is_accumulating
        assert trainer.device == torch.device("cpu")

    def test_train_one_epoch(self, tmp_path: Path) -> None:
        model = self._make_tiny_model()
        loader = self._make_loader()
        config = TrainConfig(
            epochs=1, checkpoint_dir=str(tmp_path),
            mixed_precision=False, warmup_epochs=0,
        )

        trainer = ColabTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
            accumulation_steps=1,
        )

        history = trainer.train()
        assert len(history["train_loss"]) == 1
        assert len(history["species_f1"]) == 1
        assert len(history["avg_f1"]) == 1

    def test_train_with_accumulation(self, tmp_path: Path) -> None:
        model = self._make_tiny_model()
        loader = self._make_loader(n=16, batch_size=4)
        config = TrainConfig(
            epochs=1, checkpoint_dir=str(tmp_path),
            mixed_precision=False, warmup_epochs=0,
        )

        trainer = ColabTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
            accumulation_steps=2,  # effective batch = 8
        )

        history = trainer.train()
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_crash_recovery_roundtrip(self, tmp_path: Path) -> None:
        model = self._make_tiny_model()
        loader = self._make_loader()
        config = TrainConfig(
            epochs=2, checkpoint_dir=str(tmp_path),
            mixed_precision=False, warmup_epochs=0,
        )

        # Train 1 epoch
        trainer = ColabTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
            crash_interval_minutes=0,  # Save immediately
        )
        # Manually force a crash checkpoint
        trainer.crash_ckpt._last_save_time = 0
        trainer.crash_ckpt.maybe_save(
            model, trainer.optimizer, trainer.scheduler,
            epoch=1, batch_idx=0,
        )

        # New trainer should detect recovery
        trainer2 = ColabTrainer(
            model=self._make_tiny_model(),
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
        )
        recovered = trainer2.resume_from_crash()
        assert recovered

    def test_gpu_monitor_in_trainer(self, tmp_path: Path) -> None:
        model = self._make_tiny_model()
        loader = self._make_loader()
        config = TrainConfig(
            epochs=1, checkpoint_dir=str(tmp_path),
            mixed_precision=False, warmup_epochs=0,
        )

        trainer = ColabTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
        )
        history = trainer.train()
        assert "gpu_peak_mb" in history


# Need to import TrainConfig here since it's used above
from phytoveda.training.trainer import TrainConfig

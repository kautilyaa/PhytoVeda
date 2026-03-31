"""Tests for the Colab integration module.

Tests DriveManager path resolution, directory scaffolding, sync utilities,
and ColabEnvironment detection. Uses tmp_path fixture to simulate SSD/Drive
paths without requiring actual Google Colab.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phytoveda.colab.drive import DriveManager, _human_size
from phytoveda.colab.environment import ColabEnvironment


# ─── DriveManager Path Resolution ────────────────────────────────────────


class TestDriveManagerPaths:
    def test_default_ssd_base(self) -> None:
        dm = DriveManager()
        assert dm.ssd_base == Path("/content")

    def test_default_drive_base(self) -> None:
        dm = DriveManager()
        assert dm.drive_base == Path("/content/drive/MyDrive/PhytoVeda")

    def test_custom_project_name(self) -> None:
        dm = DriveManager(drive_project="MyPlantProject")
        assert dm.drive_base == Path("/content/drive/MyDrive/MyPlantProject")

    def test_custom_ssd_root(self) -> None:
        dm = DriveManager(ssd_root="/tmp/test_ssd")
        assert dm.ssd_base == Path("/tmp/test_ssd")
        assert dm.datasets_dir == Path("/tmp/test_ssd/datasets")

    def test_custom_mount_point(self) -> None:
        dm = DriveManager(drive_mount_point="/mnt/gdrive")
        assert dm.drive_base == Path("/mnt/gdrive/MyDrive/PhytoVeda")

    def test_datasets_on_ssd(self) -> None:
        dm = DriveManager()
        assert str(dm.datasets_dir).startswith("/content")
        assert "drive" not in str(dm.datasets_dir)

    def test_cache_on_ssd(self) -> None:
        dm = DriveManager()
        assert str(dm.cache_dir).startswith("/content")

    def test_tmp_on_ssd(self) -> None:
        dm = DriveManager()
        assert str(dm.tmp_dir).startswith("/content")

    def test_checkpoints_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.checkpoints_dir)
        assert str(dm.checkpoints_dir).endswith("checkpoints")

    def test_results_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.results_dir)

    def test_chromadb_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.chromadb_dir)

    def test_quarantine_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.quarantine_dir)

    def test_logs_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.logs_dir)

    def test_configs_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.configs_dir)

    def test_reports_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.reports_dir)

    def test_expert_queue_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.expert_queue_dir)

    def test_traceability_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.traceability_dir)

    def test_ledger_path(self) -> None:
        dm = DriveManager()
        assert dm.ledger_path.name == "events.jsonl"
        assert "traceability" in str(dm.ledger_path)

    def test_texts_on_drive(self) -> None:
        dm = DriveManager()
        assert "drive" in str(dm.texts_dir)

    def test_all_ssd_dirs(self) -> None:
        dm = DriveManager()
        assert len(dm._ssd_dirs) == 3
        for d in dm._ssd_dirs:
            assert "drive" not in str(d)

    def test_all_drive_dirs(self) -> None:
        dm = DriveManager()
        assert len(dm._drive_dirs) == 10
        for d in dm._drive_dirs:
            assert "drive" in str(d)


# ─── DriveManager Scaffold ───────────────────────────────────────────────


class TestDriveManagerScaffold:
    def test_scaffold_creates_ssd_dirs(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        result = dm.scaffold()
        assert len(result["ssd"]) == 3
        assert dm.datasets_dir.exists()
        assert dm.cache_dir.exists()
        assert dm.tmp_dir.exists()

    def test_scaffold_creates_drive_dirs(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        result = dm.scaffold()
        assert len(result["drive"]) == 10
        assert dm.checkpoints_dir.exists()
        assert dm.results_dir.exists()
        assert dm.chromadb_dir.exists()
        assert dm.quarantine_dir.exists()
        assert dm.logs_dir.exists()
        assert dm.configs_dir.exists()
        assert dm.reports_dir.exists()
        assert dm.expert_queue_dir.exists()
        assert dm.traceability_dir.exists()
        assert dm.texts_dir.exists()

    def test_scaffold_idempotent(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()
        # Should not raise on second call
        result = dm.scaffold()
        assert dm.datasets_dir.exists()


# ─── DriveManager Sync ───────────────────────────────────────────────────


class TestDriveManagerSync:
    def test_sync_file_to_drive(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        # Create a file on "SSD"
        src = dm.datasets_dir / "test.txt"
        src.write_text("hello")

        # Sync to Drive
        dest = dm.sync_to_drive(src, "results")
        assert dest.exists()
        assert dest.read_text() == "hello"
        assert "drive" in str(dest)

    def test_sync_directory_to_drive(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        # Create a directory on "SSD"
        src_dir = dm.cache_dir / "subdir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("aaa")
        (src_dir / "b.txt").write_text("bbb")

        dest = dm.sync_to_drive(src_dir, "results")
        assert dest.is_dir()
        assert (dest / "a.txt").read_text() == "aaa"
        assert (dest / "b.txt").read_text() == "bbb"

    def test_sync_from_drive(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        # Put a file on "Drive"
        drive_file = dm.checkpoints_dir / "model.pt"
        drive_file.write_text("weights")

        # Sync to SSD
        dest = dm.sync_from_drive("checkpoints/model.pt", dm.cache_dir / "model.pt")
        assert dest.exists()
        assert dest.read_text() == "weights"

    def test_sync_from_drive_not_found(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        with pytest.raises(FileNotFoundError):
            dm.sync_from_drive("nonexistent/file.txt", dm.cache_dir / "out.txt")

    def test_sync_overwrites_existing(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        src = dm.datasets_dir / "data.csv"
        src.write_text("version1")
        dm.sync_to_drive(src, "results")

        src.write_text("version2")
        dest = dm.sync_to_drive(src, "results")
        assert dest.read_text() == "version2"


# ─── DriveManager Reports ───────────────────────────────────────────────


class TestDriveManagerReports:
    def test_save_report(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        path = dm.save_report("# Neem Report\nHealthy", "neem_report.md")
        assert path.exists()
        assert path.read_text() == "# Neem Report\nHealthy"
        assert "reports" in str(path)

    def test_save_json_report(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        data = {"species": "Azadirachta indica", "status": "Healthy"}
        path = dm.save_report(json.dumps(data), "neem.json")
        loaded = json.loads(path.read_text())
        assert loaded["species"] == "Azadirachta indica"


# ─── DriveManager Checkpoints ───────────────────────────────────────────


class TestDriveManagerCheckpoints:
    def test_list_checkpoints_empty(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()
        assert dm.list_checkpoints() == []

    def test_list_checkpoints(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        (dm.checkpoints_dir / "epoch_01.pt").write_text("a")
        (dm.checkpoints_dir / "epoch_02.pt").write_text("b")
        (dm.checkpoints_dir / "readme.txt").write_text("not a checkpoint")

        ckpts = dm.list_checkpoints()
        assert len(ckpts) == 2
        assert all(p.suffix == ".pt" for p in ckpts)

    def test_latest_checkpoint_best(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        (dm.checkpoints_dir / "best_model.pt").write_text("best")
        (dm.checkpoints_dir / "epoch_05.pt").write_text("other")

        assert dm.latest_checkpoint() == dm.checkpoints_dir / "best_model.pt"

    def test_latest_checkpoint_none(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()
        assert dm.latest_checkpoint() is None


# ─── DriveManager Utilities ──────────────────────────────────────────────


class TestDriveManagerUtils:
    def test_is_drive_mounted_false(self) -> None:
        dm = DriveManager(drive_mount_point="/nonexistent/mount")
        assert dm.is_drive_mounted is False

    def test_disk_usage(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        dm.scaffold()

        # Write some data
        (dm.datasets_dir / "img.jpg").write_bytes(b"x" * 1024)
        usage = dm.disk_usage()
        assert "datasets (SSD)" in usage
        assert "checkpoints (Drive)" in usage

    def test_summary(self, tmp_path: Path) -> None:
        dm = DriveManager(
            ssd_root=str(tmp_path / "ssd"),
            drive_mount_point=str(tmp_path / "drive"),
        )
        summary = dm.summary()
        assert "PhytoVeda Storage Layout" in summary
        assert "Datasets" in summary
        assert "Checkpoints" in summary


# ─── _human_size ─────────────────────────────────────────────────────────


class TestHumanSize:
    def test_bytes(self) -> None:
        assert _human_size(500) == "500.0 B"

    def test_kilobytes(self) -> None:
        assert _human_size(1024) == "1.0 KB"

    def test_megabytes(self) -> None:
        assert _human_size(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self) -> None:
        assert _human_size(1024 ** 3) == "1.0 GB"

    def test_zero(self) -> None:
        assert _human_size(0) == "0.0 B"


# ─── ColabEnvironment ────────────────────────────────────────────────────


class TestColabEnvironment:
    def test_is_colab_false_locally(self) -> None:
        """Outside Colab, should return False."""
        assert ColabEnvironment.is_colab() is False

    def test_python_version(self) -> None:
        ver = ColabEnvironment.python_version()
        assert "3." in ver

    def test_python_version_tuple(self) -> None:
        major, minor, micro = ColabEnvironment.python_version_tuple()
        assert major == 3
        assert minor >= 11  # We require at least 3.14 but test runs on local

    def test_device_returns_string(self) -> None:
        device = ColabEnvironment.device()
        assert device in ("cpu", "cuda", "mps")

    def test_has_gpu_returns_bool(self) -> None:
        assert isinstance(ColabEnvironment.has_gpu(), bool)

    def test_gpu_name_local(self) -> None:
        # May be None or a string depending on local GPU
        result = ColabEnvironment.gpu_name()
        assert result is None or isinstance(result, str)

    def test_gpu_memory_local(self) -> None:
        result = ColabEnvironment.gpu_memory_gb()
        assert result is None or result > 0

    def test_has_tpu_false_locally(self) -> None:
        assert ColabEnvironment.has_tpu() is False

    def test_summary(self) -> None:
        env = ColabEnvironment()
        summary = env.summary()
        assert "PhytoVeda Environment" in summary
        assert "Python" in summary
        assert "Device" in summary

    def test_check_ready(self) -> None:
        env = ColabEnvironment()
        checks = env.check_ready()
        assert "gpu_available" in checks
        assert "pkg_torch" in checks
        assert "pkg_phytoveda" in checks
        assert isinstance(checks["gpu_available"], bool)

    def test_configure_free_threading(self) -> None:
        import os
        old = os.environ.get("PYTHON_GIL")
        ColabEnvironment.configure_free_threading()
        assert os.environ["PYTHON_GIL"] == "0"
        # Restore
        if old is not None:
            os.environ["PYTHON_GIL"] = old
        else:
            del os.environ["PYTHON_GIL"]

    def test_configure_cuda_sets_defaults(self) -> None:
        import os
        # Remove if present to test setdefault
        old = os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        ColabEnvironment.configure_cuda()
        assert "CUDA_LAUNCH_BLOCKING" in os.environ
        # Restore
        if old is not None:
            os.environ["CUDA_LAUNCH_BLOCKING"] = old

    def test_set_seed_runs(self) -> None:
        """set_seed should run without errors."""
        ColabEnvironment.set_seed(123)

    def test_is_free_threaded(self) -> None:
        result = ColabEnvironment.is_free_threaded()
        assert isinstance(result, bool)

"""Google Drive path management for Colab environments.

Strategy:
    - SSD (/content/) — fast, ephemeral storage for datasets and temp files
    - Drive (/content/drive/MyDrive/<project>/) — persistent storage for everything else

All persistent artifacts (checkpoints, results, configs, chromadb, quarantine, logs)
are stored on Drive so they survive runtime restarts. Datasets are downloaded to SSD
for maximum I/O throughput during training.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DriveManager:
    """Manages the SSD + Drive path strategy for Google Colab.

    Usage:
        from phytoveda.colab import DriveManager

        dm = DriveManager()           # defaults to /content/ SSD + MyDrive/PhytoVeda
        dm.mount()                     # mounts Google Drive (interactive auth)
        dm.scaffold()                  # creates all directories

        # Use paths
        dm.datasets_dir               # /content/datasets (SSD — fast)
        dm.checkpoints_dir            # /content/drive/MyDrive/PhytoVeda/checkpoints
        dm.results_dir                # /content/drive/MyDrive/PhytoVeda/results
        dm.chromadb_dir               # /content/drive/MyDrive/PhytoVeda/chromadb
        dm.quarantine_dir             # /content/drive/MyDrive/PhytoVeda/quarantine
        dm.logs_dir                   # /content/drive/MyDrive/PhytoVeda/logs
        dm.configs_dir                # /content/drive/MyDrive/PhytoVeda/configs
        dm.reports_dir                # /content/drive/MyDrive/PhytoVeda/reports
        dm.expert_queue_dir           # /content/drive/MyDrive/PhytoVeda/expert_queue
        dm.ledger_path                # /content/drive/MyDrive/PhytoVeda/traceability/events.jsonl

        # Sync a file from SSD to Drive
        dm.sync_to_drive(ssd_file, drive_subpath)
    """

    drive_project: str = "PhytoVeda"
    ssd_root: str = "/content"
    drive_mount_point: str = "/content/drive"
    _mounted: bool = field(default=False, repr=False)

    # ─── Core path roots ─────────────────────────────────────────────────

    @property
    def ssd_base(self) -> Path:
        """Root of the fast ephemeral SSD storage."""
        return Path(self.ssd_root)

    @property
    def drive_base(self) -> Path:
        """Root of the persistent Drive storage for this project."""
        return Path(self.drive_mount_point) / "MyDrive" / self.drive_project

    # ─── SSD paths (ephemeral, fast I/O) ─────────────────────────────────

    @property
    def datasets_dir(self) -> Path:
        """Downloaded datasets — SSD for fast training I/O."""
        return self.ssd_base / "datasets"

    @property
    def cache_dir(self) -> Path:
        """Temporary processing cache — SSD."""
        return self.ssd_base / "cache"

    @property
    def tmp_dir(self) -> Path:
        """Scratch space — SSD."""
        return self.ssd_base / "tmp"

    # ─── Drive paths (persistent across sessions) ────────────────────────

    @property
    def checkpoints_dir(self) -> Path:
        """Model checkpoints — persisted to Drive."""
        return self.drive_base / "checkpoints"

    @property
    def results_dir(self) -> Path:
        """Evaluation results, metrics, confusion matrices."""
        return self.drive_base / "results"

    @property
    def chromadb_dir(self) -> Path:
        """ChromaDB vector store for RAG."""
        return self.drive_base / "chromadb"

    @property
    def quarantine_dir(self) -> Path:
        """Active learning quarantine images + manifest."""
        return self.drive_base / "quarantine"

    @property
    def logs_dir(self) -> Path:
        """Training logs and WandB artifacts."""
        return self.drive_base / "logs"

    @property
    def configs_dir(self) -> Path:
        """Saved training configurations."""
        return self.drive_base / "configs"

    @property
    def reports_dir(self) -> Path:
        """Generated botanical reports (JSON + Markdown)."""
        return self.drive_base / "reports"

    @property
    def expert_queue_dir(self) -> Path:
        """Human expert labeling queue."""
        return self.drive_base / "expert_queue"

    @property
    def traceability_dir(self) -> Path:
        """Supply chain traceability data."""
        return self.drive_base / "traceability"

    @property
    def ledger_path(self) -> Path:
        """Event ledger JSONL file for traceability."""
        return self.traceability_dir / "events.jsonl"

    @property
    def texts_dir(self) -> Path:
        """Ayurvedic source texts for RAG indexing — persisted on Drive."""
        return self.drive_base / "texts"

    # ─── All managed directories ─────────────────────────────────────────

    @property
    def _ssd_dirs(self) -> list[Path]:
        return [self.datasets_dir, self.cache_dir, self.tmp_dir]

    @property
    def _drive_dirs(self) -> list[Path]:
        return [
            self.checkpoints_dir,
            self.results_dir,
            self.chromadb_dir,
            self.quarantine_dir,
            self.logs_dir,
            self.configs_dir,
            self.reports_dir,
            self.expert_queue_dir,
            self.traceability_dir,
            self.texts_dir,
        ]

    # ─── Operations ──────────────────────────────────────────────────────

    def mount(self, force_remount: bool = False) -> None:
        """Mount Google Drive (only works inside Colab).

        Args:
            force_remount: Remount even if already mounted.
        """
        if self._mounted and not force_remount:
            return

        from google.colab import drive  # type: ignore[import-untyped]

        drive.mount(self.drive_mount_point, force_remount=force_remount)
        self._mounted = True

    @property
    def is_drive_mounted(self) -> bool:
        """Check if Google Drive is currently mounted."""
        return (Path(self.drive_mount_point) / "MyDrive").exists()

    def scaffold(self) -> dict[str, list[str]]:
        """Create all managed directories on both SSD and Drive.

        Returns:
            Dict with 'ssd' and 'drive' keys listing created directory paths.
        """
        created: dict[str, list[str]] = {"ssd": [], "drive": []}

        for d in self._ssd_dirs:
            d.mkdir(parents=True, exist_ok=True)
            created["ssd"].append(str(d))

        for d in self._drive_dirs:
            d.mkdir(parents=True, exist_ok=True)
            created["drive"].append(str(d))

        return created

    def sync_to_drive(self, src: str | Path, drive_subdir: str = "") -> Path:
        """Copy a file or directory from SSD to Drive.

        Args:
            src: Source path (typically on SSD).
            drive_subdir: Subdirectory under drive_base to copy into.
                         Defaults to drive_base root.

        Returns:
            Destination path on Drive.
        """
        src = Path(src)
        dest_dir = self.drive_base / drive_subdir if drive_subdir else self.drive_base
        dest_dir.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            dest = dest_dir / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            dest = dest_dir / src.name
            shutil.copy2(src, dest)

        return dest

    def sync_from_drive(self, drive_subpath: str, dest: str | Path) -> Path:
        """Copy a file or directory from Drive to a local path (typically SSD).

        Args:
            drive_subpath: Path relative to drive_base.
            dest: Destination path (typically on SSD).

        Returns:
            Destination path.
        """
        src = self.drive_base / drive_subpath
        dest = Path(dest)

        if not src.exists():
            raise FileNotFoundError(f"Drive path does not exist: {src}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)

        return dest

    def save_report(self, report_text: str, filename: str) -> Path:
        """Save a botanical report to the reports directory on Drive.

        Args:
            report_text: Report content (JSON or Markdown).
            filename: Filename (e.g., "neem_report.md").

        Returns:
            Path to saved report file.
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        dest = self.reports_dir / filename
        dest.write_text(report_text, encoding="utf-8")
        return dest

    def list_checkpoints(self) -> list[Path]:
        """List all saved model checkpoints on Drive."""
        if not self.checkpoints_dir.exists():
            return []
        return sorted(
            self.checkpoints_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def latest_checkpoint(self) -> Path | None:
        """Get the most recently saved checkpoint."""
        best = self.checkpoints_dir / "best_model.pt"
        if best.exists():
            return best
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def disk_usage(self) -> dict[str, str]:
        """Report disk usage for managed directories.

        Returns:
            Dict mapping directory names to human-readable sizes.
        """
        usage: dict[str, str] = {}
        all_dirs = [
            ("datasets (SSD)", self.datasets_dir),
            ("cache (SSD)", self.cache_dir),
            ("checkpoints (Drive)", self.checkpoints_dir),
            ("results (Drive)", self.results_dir),
            ("chromadb (Drive)", self.chromadb_dir),
            ("quarantine (Drive)", self.quarantine_dir),
            ("logs (Drive)", self.logs_dir),
            ("reports (Drive)", self.reports_dir),
            ("texts (Drive)", self.texts_dir),
        ]
        for name, path in all_dirs:
            if path.exists():
                total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                usage[name] = _human_size(total)
            else:
                usage[name] = "—"
        return usage

    def summary(self) -> str:
        """Print a human-readable summary of the storage layout."""
        lines = [
            "PhytoVeda Storage Layout",
            "=" * 50,
            "",
            "SSD (fast, ephemeral):",
            f"  Datasets:    {self.datasets_dir}",
            f"  Cache:       {self.cache_dir}",
            f"  Temp:        {self.tmp_dir}",
            "",
            "Google Drive (persistent):",
            f"  Checkpoints: {self.checkpoints_dir}",
            f"  Results:     {self.results_dir}",
            f"  ChromaDB:    {self.chromadb_dir}",
            f"  Quarantine:  {self.quarantine_dir}",
            f"  Logs:        {self.logs_dir}",
            f"  Configs:     {self.configs_dir}",
            f"  Reports:     {self.reports_dir}",
            f"  Expert Q:    {self.expert_queue_dir}",
            f"  Traceability:{self.traceability_dir}",
            f"  Texts:       {self.texts_dir}",
            "",
            f"Drive mounted: {self.is_drive_mounted}",
        ]

        ckpts = self.list_checkpoints()
        if ckpts:
            lines.append(f"Checkpoints:   {len(ckpts)} saved")
            lines.append(f"  Latest:      {ckpts[0].name}")

        return "\n".join(lines)


def _human_size(nbytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"

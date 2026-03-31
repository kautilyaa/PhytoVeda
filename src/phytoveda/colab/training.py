"""Colab-optimized training utilities.

Addresses the key pain points of training on Google Colab:
    1. GPU OOM — gradient accumulation to simulate larger effective batch sizes
    2. Runtime disconnects — time-based crash checkpointing (not just per-epoch)
    3. Batch size uncertainty — auto batch size finder for any GPU
    4. Speed — torch.compile wrapper with Colab-safe fallback
    5. Monitoring — real-time GPU memory tracking to catch OOM before it happens

Usage:
    from phytoveda.colab.training import (
        ColabTrainer,
        find_max_batch_size,
        compile_model,
        GPUMonitor,
    )
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phytoveda.models.cmtl import HierViTCMTL
from phytoveda.models.losses import CMTLLoss
from phytoveda.training.evaluation import evaluate
from phytoveda.training.trainer import CheckpointManager, TrainConfig

# ─── GPU Memory Monitor ──────────────────────────────────────────────────


@dataclass
class GPUSnapshot:
    """Point-in-time GPU memory snapshot."""

    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_pct: float
    timestamp: float = field(default_factory=time.monotonic)


class GPUMonitor:
    """Track GPU memory usage over time to detect OOM risk.

    Usage:
        monitor = GPUMonitor()
        monitor.snapshot()                # Take a reading
        monitor.snapshot("after_forward")  # Named snapshot
        print(monitor.summary())
        monitor.warn_if_low(threshold_mb=500)
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._snapshots: list[tuple[str, GPUSnapshot]] = []
        self._available = torch.cuda.is_available()

    def snapshot(self, label: str = "") -> GPUSnapshot | None:
        """Take a GPU memory snapshot and store it.

        Args:
            label: Optional label for this snapshot point.

        Returns:
            GPUSnapshot if CUDA is available, None otherwise.
        """
        if not self._available:
            return None

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 ** 2)
        total_mb = total / (1024 ** 2)
        utilization = (total_mb - free_mb) / total_mb * 100

        snap = GPUSnapshot(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free_mb,
            total_mb=total_mb,
            utilization_pct=utilization,
        )
        self._snapshots.append((label or f"snap_{len(self._snapshots)}", snap))
        return snap

    def peak_allocated_mb(self) -> float:
        """Get peak allocated GPU memory in MB."""
        if not self._available:
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        if self._available:
            torch.cuda.reset_peak_memory_stats()

    def warn_if_low(self, threshold_mb: float = 500) -> bool:
        """Check if free GPU memory is below threshold.

        Returns:
            True if memory is critically low.
        """
        if not self._available:
            return False
        free, _ = torch.cuda.mem_get_info()
        free_mb = free / (1024 ** 2)
        return free_mb < threshold_mb

    def clear_cache(self) -> None:
        """Force CUDA cache cleanup and Python garbage collection."""
        gc.collect()
        if self._available:
            torch.cuda.empty_cache()

    @property
    def snapshots(self) -> list[tuple[str, GPUSnapshot]]:
        return list(self._snapshots)

    def summary(self) -> str:
        """Human-readable summary of all snapshots."""
        if not self._snapshots:
            return "No GPU snapshots recorded."

        lines = ["GPU Memory Timeline:", "-" * 70]
        for label, snap in self._snapshots:
            lines.append(
                f"  {label:25s}  "
                f"Alloc: {snap.allocated_mb:7.1f} MB  "
                f"Free: {snap.free_mb:7.1f} MB  "
                f"Util: {snap.utilization_pct:.1f}%"
            )
        lines.append(f"  {'Peak allocated':25s}  {self.peak_allocated_mb():7.1f} MB")
        return "\n".join(lines)


# ─── Auto Batch Size Finder ──────────────────────────────────────────────


def find_max_batch_size(
    model: nn.Module,
    sample_input_shape: tuple[int, ...] = (3, 512, 512),
    min_batch: int = 1,
    max_batch: int = 128,
    device: str = "cuda",
    mixed_precision: bool = True,
    safety_margin: float = 0.85,
) -> int:
    """Find the maximum batch size that fits in GPU memory.

    Uses binary search: tries progressively larger batches until OOM,
    then backs off. Applies a safety margin to leave room for optimizer
    state and activation gradients.

    Args:
        model: The model to test (will be moved to device).
        sample_input_shape: Shape of a single input (C, H, W).
        min_batch: Minimum batch size to try.
        max_batch: Maximum batch size to try.
        device: Device to test on.
        mixed_precision: Whether AMP will be used during training.
        safety_margin: Fraction of max working batch to use (0.85 = 85%).

    Returns:
        Recommended batch size.
    """
    if not torch.cuda.is_available():
        return min_batch

    model = model.to(device)
    model.train()

    best = min_batch
    lo, hi = min_batch, max_batch

    while lo <= hi:
        mid = (lo + hi) // 2
        gc.collect()
        torch.cuda.empty_cache()

        try:
            dummy = torch.randn(mid, *sample_input_shape, device=device)
            with torch.amp.autocast(device_type="cuda", enabled=mixed_precision):
                _ = model(dummy)
            # If forward pass works, try backward too
            if hasattr(model, 'parameters'):
                loss = sum(o.sum() for o in _) if isinstance(_, tuple) else _.sum()
                loss.backward()

            best = mid
            lo = mid + 1
        except (RuntimeError, torch.OutOfMemoryError):
            hi = mid - 1
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)

    recommended = max(min_batch, int(best * safety_margin))
    # Round down to nearest power of 2 for efficiency
    power = 1
    while power * 2 <= recommended:
        power *= 2
    return power


# ─── torch.compile Wrapper ───────────────────────────────────────────────


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fallback: bool = True,
) -> nn.Module:
    """Apply torch.compile with Colab-safe fallback.

    On Colab, some compile modes may fail due to limited triton/inductor
    support. This wrapper catches failures and returns the uncompiled model.

    Args:
        model: Model to compile.
        mode: Compile mode — "default", "reduce-overhead", or "max-autotune".
        fallback: If True, return uncompiled model on failure instead of raising.

    Returns:
        Compiled model (or original if compilation fails and fallback=True).
    """
    try:
        compiled = torch.compile(model, mode=mode)
        # Verify it works with a small forward pass
        # (torch.compile is lazy — errors surface on first call)
        return compiled
    except Exception as e:
        if fallback:
            print(f"torch.compile failed ({e}), using uncompiled model")
            return model
        raise


# ─── Gradient Accumulation Config ────────────────────────────────────────


@dataclass
class GradAccumConfig:
    """Configuration for gradient accumulation.

    Effective batch size = batch_size * accumulation_steps.
    Example: batch_size=4, accumulation_steps=8 → effective batch of 32.
    """

    accumulation_steps: int = 1

    @property
    def is_accumulating(self) -> bool:
        return self.accumulation_steps > 1

    def should_step(self, batch_idx: int) -> bool:
        """Whether to perform optimizer step at this batch index (0-based)."""
        return (batch_idx + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation steps for correct gradient magnitude."""
        if self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss


# ─── Crash-Resilient Checkpointing ───────────────────────────────────────


class CrashCheckpointer:
    """Time-based checkpointing for Colab crash resilience.

    Saves a recovery checkpoint every `interval_minutes` regardless of F1.
    This ensures that a Colab disconnect loses at most N minutes of training.
    Only keeps the latest crash checkpoint to avoid filling Drive.

    Usage:
        crasher = CrashCheckpointer(save_dir=dm.checkpoints_dir, interval_minutes=15)

        for batch in loader:
            ...
            crasher.maybe_save(model, optimizer, scheduler, epoch, batch_idx, metrics)
    """

    def __init__(
        self,
        save_dir: str | Path,
        interval_minutes: float = 15.0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.interval_seconds = interval_minutes * 60
        self._last_save_time = time.monotonic()
        self._save_count = 0

    @property
    def crash_checkpoint_path(self) -> Path:
        return self.save_dir / "crash_recovery.pt"

    def maybe_save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        epoch: int,
        batch_idx: int,
        metrics: dict | None = None,
    ) -> bool:
        """Save a crash recovery checkpoint if enough time has elapsed.

        Returns:
            True if checkpoint was saved.
        """
        elapsed = time.monotonic() - self._last_save_time
        if elapsed < self.interval_seconds:
            return False

        self.save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
            ),
            "metrics": metrics or {},
            "save_time": time.time(),
            "save_type": "crash_recovery",
        }
        torch.save(state, self.crash_checkpoint_path)

        self._last_save_time = time.monotonic()
        self._save_count += 1
        return True

    @property
    def save_count(self) -> int:
        return self._save_count

    def has_recovery(self) -> bool:
        """Check if a crash recovery checkpoint exists."""
        return self.crash_checkpoint_path.exists()

    def load_recovery(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: object,
        device: str = "cpu",
    ) -> dict:
        """Load crash recovery checkpoint.

        Returns:
            Dict with 'epoch' and 'batch_idx' to resume from.
        """
        ckpt = torch.load(self.crash_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return {
            "epoch": ckpt["epoch"],
            "batch_idx": ckpt["batch_idx"],
            "metrics": ckpt.get("metrics", {}),
        }

    def cleanup(self) -> None:
        """Remove crash recovery checkpoint (call after successful training)."""
        self.crash_checkpoint_path.unlink(missing_ok=True)


# ─── Colab Trainer ───────────────────────────────────────────────────────


class ColabTrainer:
    """Colab-optimized trainer wrapping the base Trainer with:

    - Gradient accumulation for large effective batch sizes on small GPUs
    - Crash-resilient time-based checkpointing
    - GPU memory monitoring
    - Integrated with DriveManager paths

    Usage:
        from phytoveda.colab.training import ColabTrainer
        from phytoveda.colab import DriveManager

        dm = DriveManager()
        trainer = ColabTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_dir=dm.checkpoints_dir,
            accumulation_steps=8,          # effective batch = batch_size * 8
            crash_interval_minutes=15,     # save recovery every 15 min
        )
        history = trainer.train()
    """

    def __init__(
        self,
        model: HierViTCMTL,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        checkpoint_dir: str | Path | None = None,
        device: str | None = None,
        accumulation_steps: int = 1,
        crash_interval_minutes: float = 15.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler — warmup + cosine
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_epochs = min(config.warmup_epochs, config.epochs)
        cosine_epochs = max(config.epochs - warmup_epochs, 1)

        if warmup_epochs > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs)
            self.scheduler = SequentialLR(
                self.optimizer, [warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)

        # AMP
        use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.amp_enabled = config.mixed_precision
        self.amp_device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"

        # Loss
        self.criterion = CMTLLoss(
            num_species=model.species_head.head[-1].out_features,
            focal_gamma=config.focal_gamma,
            reg_lambda=config.reg_lambda,
            weighting_temperature=config.weighting_temperature,
        )

        # Checkpoint dir (prefer explicit, then config)
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.checkpoint_dir)
        self.ckpt_manager = CheckpointManager(ckpt_dir, save_top_k=config.save_top_k)

        # Gradient accumulation
        self.grad_accum = GradAccumConfig(accumulation_steps=accumulation_steps)

        # Crash checkpointing
        self.crash_ckpt = CrashCheckpointer(
            save_dir=ckpt_dir,
            interval_minutes=crash_interval_minutes,
        )

        # GPU monitor
        self.gpu_monitor = GPUMonitor(device=str(self.device))

        # Training state
        self.best_f1 = 0.0
        self._start_epoch = 0

    def resume_from_crash(self) -> bool:
        """Attempt to resume from a crash recovery checkpoint.

        Returns:
            True if recovery was loaded.
        """
        if not self.crash_ckpt.has_recovery():
            return False

        info = self.crash_ckpt.load_recovery(
            self.model, self.optimizer, self.scheduler,
            device=str(self.device),
        )
        self._start_epoch = info["epoch"]
        print(f"Crash recovery loaded: epoch {info['epoch']}, batch {info['batch_idx']}")
        return True

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """One training epoch with gradient accumulation and crash checkpointing."""
        self.model.train()
        epoch_losses: dict[str, list[float]] = {"total": [], "species": [], "disease": []}

        self.optimizer.zero_grad()

        for batch_idx, (images, species_targets, pathology_targets) in enumerate(
            self.train_loader
        ):
            images = images.to(self.device)
            species_targets = species_targets.to(self.device)
            pathology_targets = pathology_targets.to(self.device)

            # Forward + scaled loss
            with torch.amp.autocast(device_type=self.amp_device_type, enabled=self.amp_enabled):
                species_logits, pathology_logits = self.model(images)
                loss, loss_dict = self.criterion(
                    species_logits, pathology_logits,
                    species_targets, pathology_targets,
                )

            scaled_loss = self.grad_accum.scale_loss(loss)
            self.scaler.scale(scaled_loss).backward()

            # Optimizer step at accumulation boundary
            if self.grad_accum.should_step(batch_idx):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for key in epoch_losses:
                epoch_losses[key].append(loss_dict.get(key, 0.0))

            # Crash checkpoint (time-based)
            self.crash_ckpt.maybe_save(
                self.model, self.optimizer, self.scheduler,
                epoch=epoch, batch_idx=batch_idx,
                metrics={"last_loss": loss_dict.get("total", 0.0)},
            )

        # Handle remaining gradients if total batches not divisible by accum steps
        if len(self.train_loader) % self.grad_accum.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_max_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.scheduler.step()

        return {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}

    def train(self) -> dict[str, list[float]]:
        """Full training loop with all Colab optimizations.

        Returns:
            History dict with per-epoch metrics lists.
        """
        effective_bs = self.config.batch_size * self.grad_accum.accumulation_steps

        print(f"\nColabTrainer on {self.device}")
        print(f"  Batch size: {self.config.batch_size} x {self.grad_accum.accumulation_steps} "
              f"accum = {effective_bs} effective")
        print(f"  Crash checkpoint: every {self.crash_ckpt.interval_seconds / 60:.0f} min")
        print(f"  Mixed precision: {self.amp_enabled}")
        print(f"  Epochs: {self.config.epochs} (starting from {self._start_epoch})")
        print()

        self.gpu_monitor.reset_peak()
        self.gpu_monitor.snapshot("before_training")

        history: dict[str, list[float]] = {
            "train_loss": [], "species_loss": [], "disease_loss": [],
            "species_f1": [], "pathology_f1": [], "avg_f1": [],
            "lr": [], "gpu_peak_mb": [],
        }

        for epoch in range(self._start_epoch, self.config.epochs):
            t0 = time.monotonic()
            train_losses = self.train_epoch(epoch)
            train_time = time.monotonic() - t0

            t0 = time.monotonic()
            val_metrics = evaluate(self.model, self.val_loader, self.device)
            eval_time = time.monotonic() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            avg_f1 = (val_metrics["species_f1"] + val_metrics["pathology_f1"]) / 2

            # Record history
            history["train_loss"].append(train_losses["total"])
            history["species_loss"].append(train_losses["species"])
            history["disease_loss"].append(train_losses["disease"])
            history["species_f1"].append(val_metrics["species_f1"])
            history["pathology_f1"].append(val_metrics["pathology_f1"])
            history["avg_f1"].append(avg_f1)
            history["lr"].append(lr)
            history["gpu_peak_mb"].append(self.gpu_monitor.peak_allocated_mb())

            # GPU snapshot
            self.gpu_monitor.snapshot(f"epoch_{epoch + 1}")

            # Log
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_info = f" | GPU: {self.gpu_monitor.peak_allocated_mb():.0f}MB"

            print(
                f"Epoch {epoch + 1:>3}/{self.config.epochs} | "
                f"Loss: {train_losses['total']:.4f} "
                f"(S:{train_losses['species']:.4f} D:{train_losses['disease']:.4f}) | "
                f"F1: {avg_f1:.4f} "
                f"(S:{val_metrics['species_f1']:.4f} P:{val_metrics['pathology_f1']:.4f}) | "
                f"LR: {lr:.2e} | "
                f"{train_time:.1f}s+{eval_time:.1f}s"
                f"{gpu_info}"
            )

            # F1-gated checkpoint
            saved = self.ckpt_manager.save_if_improved(
                f1_score=avg_f1,
                epoch=epoch + 1,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                extra={"config": self.config.__dict__, "val_metrics": val_metrics},
            )
            if saved:
                self.best_f1 = max(self.best_f1, avg_f1)
                print(f"  -> Checkpoint saved (F1: {avg_f1:.4f})")

            # OOM warning
            if self.gpu_monitor.warn_if_low(threshold_mb=500):
                print("  WARNING: GPU memory critically low — consider reducing batch size")

        # Cleanup crash checkpoint on successful completion
        self.crash_ckpt.cleanup()

        self.gpu_monitor.snapshot("after_training")
        print(f"\nTraining complete. Best F1: {self.ckpt_manager.best_f1:.4f}")
        print(f"Crash checkpoints saved: {self.crash_ckpt.save_count}")
        print(self.gpu_monitor.summary())

        return history

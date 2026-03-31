"""Training pipeline with dynamic task weighting, AMP, and F1-gated checkpointing.

Usage:
    python -m phytoveda.training.trainer --config configs/hiervit_cmtl.yaml --data-root data

Or via the installed entry point:
    phytoveda-train --config configs/hiervit_cmtl.yaml --data-root data
"""

from __future__ import annotations

import argparse
import heapq
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from phytoveda.data.datasets import build_dataloaders
from phytoveda.models.cmtl import HierViTCMTL
from phytoveda.models.losses import CMTLLoss
from phytoveda.training.evaluation import evaluate


@dataclass
class TrainConfig:
    """Training hyperparameters — loaded from YAML config."""

    # Model
    backbone: str = "vit_huge_patch14_dinov2.lvd142m"
    pretrained: bool = True
    image_size: int = 512
    species_hidden_dim: int = 1024
    pathology_hidden_dim: int = 512
    dropout: float = 0.3
    num_species: int = 147
    num_pathologies: int = 8

    # Training
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 16
    weight_decay: float = 0.01
    grad_clip_max_norm: float = 1.0
    warmup_epochs: int = 5
    mixed_precision: bool = True

    # Loss
    focal_gamma: float = 2.0
    reg_lambda: float = 0.01
    weighting_temperature: float = 2.0

    # Data
    data_root: str = "data"
    datasets: list[str] = field(
        default_factory=lambda: ["herbify", "assam", "medleafx", "cimpd", "simp", "earlynsd"]
    )
    num_workers: int = 8

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from YAML file, mapping nested structure to flat fields."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()

        if model := raw.get("model"):
            config.backbone = model.get("backbone", config.backbone)
            config.pretrained = model.get("pretrained", config.pretrained)
            config.image_size = model.get("image_size", config.image_size)
            config.species_hidden_dim = model.get("species_hidden_dim", config.species_hidden_dim)
            config.pathology_hidden_dim = model.get(
                "pathology_hidden_dim", config.pathology_hidden_dim
            )
            config.dropout = model.get("dropout", config.dropout)

        if species := raw.get("species"):
            config.num_species = species.get("num_classes", config.num_species)
        if pathology := raw.get("pathology"):
            config.num_pathologies = pathology.get("num_classes", config.num_pathologies)

        if loss := raw.get("loss"):
            config.focal_gamma = loss.get("focal_gamma", config.focal_gamma)
            config.reg_lambda = loss.get("reg_lambda", config.reg_lambda)
            config.weighting_temperature = loss.get(
                "weighting_temperature", config.weighting_temperature
            )

        if training := raw.get("training"):
            config.epochs = training.get("epochs", config.epochs)
            config.batch_size = training.get("batch_size", config.batch_size)
            config.learning_rate = training.get("learning_rate", config.learning_rate)
            config.weight_decay = training.get("weight_decay", config.weight_decay)
            config.grad_clip_max_norm = training.get(
                "grad_clip_max_norm", config.grad_clip_max_norm
            )
            config.warmup_epochs = training.get("warmup_epochs", config.warmup_epochs)
            config.mixed_precision = training.get("mixed_precision", config.mixed_precision)

        if data := raw.get("data"):
            config.num_workers = data.get("num_workers", config.num_workers)
            config.datasets = data.get("datasets", config.datasets)

        if checkpoint := raw.get("checkpoint"):
            config.checkpoint_dir = checkpoint.get("dir", config.checkpoint_dir)
            config.save_top_k = checkpoint.get("save_top_k", config.save_top_k)

        return config


class CheckpointManager:
    """Maintains top-K best checkpoints by F1 score, deleting the worst when full."""

    def __init__(self, checkpoint_dir: Path, save_top_k: int = 3) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_top_k = save_top_k
        # Min-heap of (f1_score, path_str) — smallest F1 at top for easy eviction
        self._heap: list[tuple[float, str]] = []

    def save_if_improved(
        self,
        f1_score: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        extra: dict | None = None,
    ) -> bool:
        """Save checkpoint if F1 qualifies for top-K. Returns True if saved."""
        # Check if this qualifies
        if len(self._heap) >= self.save_top_k and f1_score <= self._heap[0][0]:
            return False

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}_f1_{f1_score:.4f}.pt"

        state = {
            "epoch": epoch,
            "f1_score": f1_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if hasattr(scheduler, "state_dict")
                else None
            ),
        }
        if extra:
            state.update(extra)

        torch.save(state, ckpt_path)

        # Also save as best_model.pt symlink/copy for easy access
        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)

        heapq.heappush(self._heap, (f1_score, str(ckpt_path)))

        # Evict worst if over budget
        if len(self._heap) > self.save_top_k:
            _, evict_path = heapq.heappop(self._heap)
            Path(evict_path).unlink(missing_ok=True)

        return True

    @property
    def best_f1(self) -> float:
        """Current best F1 across all saved checkpoints."""
        if not self._heap:
            return 0.0
        return max(f1 for f1, _ in self._heap)


class Trainer:
    """Epoch-based trainer for HierViTCMTL with mixed precision and dynamic weighting."""

    def __init__(
        self,
        model: HierViTCMTL,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Warmup + Cosine Annealing scheduler
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

        # AMP — GradScaler only useful on CUDA
        use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.amp_enabled = config.mixed_precision
        self.amp_device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"

        self.criterion = CMTLLoss(
            num_species=model.species_head.head[-1].out_features,
            focal_gamma=config.focal_gamma,
            reg_lambda=config.reg_lambda,
            weighting_temperature=config.weighting_temperature,
        )

        self.ckpt_manager = CheckpointManager(
            Path(config.checkpoint_dir), save_top_k=config.save_top_k
        )

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch with AMP and gradient clipping."""
        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "total": [],
            "species": [],
            "disease": [],
        }

        for images, species_targets, pathology_targets in self.train_loader:
            images = images.to(self.device)
            species_targets = species_targets.to(self.device)
            pathology_targets = pathology_targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.amp_device_type, enabled=self.amp_enabled):
                species_logits, pathology_logits = self.model(images)
                loss, loss_dict = self.criterion(
                    species_logits,
                    pathology_logits,
                    species_targets,
                    pathology_targets,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in epoch_losses:
                epoch_losses[key].append(loss_dict.get(key, 0.0))

        self.scheduler.step()

        return {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}

    def train(self) -> dict[str, float]:
        """Full training loop with evaluation and F1-gated checkpointing.

        Returns:
            Final validation metrics from the last epoch.
        """
        print(f"\nTraining on {self.device} for {self.config.epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"  Warmup: {self.config.warmup_epochs} epochs, LR: {self.config.learning_rate}")
        print(f"  Mixed precision: {self.amp_enabled}, Checkpoints: top-{self.config.save_top_k}")
        print()

        val_metrics: dict[str, float] = {}

        for epoch in range(self.config.epochs):
            t0 = time.monotonic()
            train_losses = self.train_epoch()
            train_time = time.monotonic() - t0

            t0 = time.monotonic()
            val_metrics = evaluate(self.model, self.val_loader, self.device)
            eval_time = time.monotonic() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            avg_f1 = (val_metrics["species_f1"] + val_metrics["pathology_f1"]) / 2

            print(
                f"Epoch {epoch + 1:>3}/{self.config.epochs} | "
                f"Loss: {train_losses['total']:.4f} "
                f"(S: {train_losses['species']:.4f}, D: {train_losses['disease']:.4f}) | "
                f"F1: {avg_f1:.4f} "
                f"(S: {val_metrics['species_f1']:.4f}, P: {val_metrics['pathology_f1']:.4f}) | "
                f"LR: {lr:.2e} | "
                f"Time: {train_time:.1f}s+{eval_time:.1f}s"
            )

            saved = self.ckpt_manager.save_if_improved(
                f1_score=avg_f1,
                epoch=epoch + 1,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                extra={"config": self.config.__dict__, "val_metrics": val_metrics},
            )
            if saved:
                print(f"  -> Checkpoint saved (F1: {avg_f1:.4f})")

        print(f"\nTraining complete. Best F1: {self.ckpt_manager.best_f1:.4f}")
        return val_metrics


def load_config(config_path: str | Path, overrides: dict | None = None) -> TrainConfig:
    """Load TrainConfig from YAML with optional CLI overrides."""
    config = TrainConfig.from_yaml(config_path)
    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
    return config


def main() -> None:
    """Entry point for training — loads config, builds pipeline, and trains."""
    parser = argparse.ArgumentParser(description="PhytoVeda Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hiervit_cmtl.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override data root directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Override checkpoint directory"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(
        args.config,
        overrides={
            "data_root": args.data_root,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_workers": args.num_workers,
            "checkpoint_dir": args.checkpoint_dir,
        },
    )

    print("PhytoVeda Training Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Backbone: {config.backbone}")
    print(f"Species classes: {config.num_species}, Pathology classes: {config.num_pathologies}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, LR: {config.learning_rate}")
    print()

    # Build data pipeline
    print("Loading datasets...")
    train_loader, val_loader, _test_loader, taxonomy = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        datasets=config.datasets,
    )

    # Update num_species from actual taxonomy
    config.num_species = taxonomy.num_species
    print(f"Taxonomy: {taxonomy.num_species} species registered\n")

    # Build model
    print("Building model...")
    model = HierViTCMTL(
        num_species=config.num_species,
        num_pathologies=config.num_pathologies,
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        image_size=config.image_size,
        species_hidden_dim=config.species_hidden_dim,
        pathology_hidden_dim=config.pathology_hidden_dim,
        dropout=config.dropout,
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable:,} trainable\n")

    # Build trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Resume from checkpoint if requested
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=trainer.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resumed from epoch {ckpt.get('epoch', '?')}, F1: {ckpt.get('f1_score', '?')}\n")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

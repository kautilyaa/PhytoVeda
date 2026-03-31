"""Evaluation metrics for dual-head CMTL model.

Computes per-task accuracy, F1-score, top-5 accuracy, precision, recall,
and confusion matrices for both species and pathology heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from phytoveda.models.cmtl import HierViTCMTL


def _top_k_accuracy(logits_list: list[torch.Tensor], targets: list[int], k: int = 5) -> float:
    """Compute top-K accuracy from accumulated logits and targets."""
    if not logits_list:
        return 0.0
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.tensor(targets)

    # Clamp k to number of classes
    k = min(k, all_logits.shape[1])
    top_k_preds = all_logits.topk(k, dim=1).indices
    correct = top_k_preds.eq(all_targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


@torch.no_grad()
def evaluate(
    model: HierViTCMTL,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on species and pathology tasks.

    Returns dict with species_f1, pathology_f1, species_acc, pathology_acc,
    species_top5_acc.
    """
    model.eval()

    all_species_preds: list[int] = []
    all_species_targets: list[int] = []
    all_pathology_preds: list[int] = []
    all_pathology_targets: list[int] = []
    species_logits_list: list[torch.Tensor] = []

    for images, species_targets, pathology_targets in dataloader:
        images = images.to(device)
        species_logits, pathology_logits = model(images)

        species_logits_list.append(species_logits.cpu())
        all_species_preds.extend(species_logits.argmax(dim=1).cpu().tolist())
        all_species_targets.extend(species_targets.tolist())
        all_pathology_preds.extend(pathology_logits.argmax(dim=1).cpu().tolist())
        all_pathology_targets.extend(pathology_targets.tolist())

    return {
        "species_f1": f1_score(
            all_species_targets, all_species_preds, average="weighted", zero_division=0
        ),
        "pathology_f1": f1_score(
            all_pathology_targets, all_pathology_preds, average="weighted", zero_division=0
        ),
        "species_acc": accuracy_score(all_species_targets, all_species_preds),
        "pathology_acc": accuracy_score(all_pathology_targets, all_pathology_preds),
        "species_top5_acc": _top_k_accuracy(species_logits_list, all_species_targets, k=5),
    }


@dataclass
class DetailedMetrics:
    """Full evaluation results with per-class breakdowns."""

    species_f1: float = 0.0
    pathology_f1: float = 0.0
    species_acc: float = 0.0
    pathology_acc: float = 0.0
    species_top5_acc: float = 0.0
    species_report: str = ""
    pathology_report: str = ""
    species_confusion: np.ndarray = field(default_factory=lambda: np.array([]))
    pathology_confusion: np.ndarray = field(default_factory=lambda: np.array([]))


@torch.no_grad()
def evaluate_detailed(
    model: HierViTCMTL,
    dataloader: DataLoader,
    device: torch.device,
    species_names: list[str] | None = None,
    pathology_names: list[str] | None = None,
) -> DetailedMetrics:
    """Full evaluation with per-class precision/recall/F1 and confusion matrices.

    Args:
        model: Trained HierViTCMTL model.
        dataloader: Validation or test DataLoader.
        device: Torch device.
        species_names: Optional list of species names for the classification report.
        pathology_names: Optional list of pathology class names.

    Returns:
        DetailedMetrics with all computed metrics.
    """
    model.eval()

    all_species_preds: list[int] = []
    all_species_targets: list[int] = []
    all_pathology_preds: list[int] = []
    all_pathology_targets: list[int] = []
    species_logits_list: list[torch.Tensor] = []

    for images, species_targets, pathology_targets in dataloader:
        images = images.to(device)
        species_logits, pathology_logits = model(images)

        species_logits_list.append(species_logits.cpu())
        all_species_preds.extend(species_logits.argmax(dim=1).cpu().tolist())
        all_species_targets.extend(species_targets.tolist())
        all_pathology_preds.extend(pathology_logits.argmax(dim=1).cpu().tolist())
        all_pathology_targets.extend(pathology_targets.tolist())

    metrics = DetailedMetrics()

    metrics.species_f1 = f1_score(
        all_species_targets, all_species_preds, average="weighted", zero_division=0
    )
    metrics.pathology_f1 = f1_score(
        all_pathology_targets, all_pathology_preds, average="weighted", zero_division=0
    )
    metrics.species_acc = accuracy_score(all_species_targets, all_species_preds)
    metrics.pathology_acc = accuracy_score(all_pathology_targets, all_pathology_preds)
    metrics.species_top5_acc = _top_k_accuracy(species_logits_list, all_species_targets, k=5)

    # Per-class reports
    metrics.species_report = classification_report(
        all_species_targets,
        all_species_preds,
        target_names=species_names,
        zero_division=0,
        output_dict=False,
    )
    metrics.pathology_report = classification_report(
        all_pathology_targets,
        all_pathology_preds,
        target_names=pathology_names,
        zero_division=0,
        output_dict=False,
    )

    # Confusion matrices
    metrics.species_confusion = confusion_matrix(all_species_targets, all_species_preds)
    metrics.pathology_confusion = confusion_matrix(all_pathology_targets, all_pathology_preds)

    return metrics

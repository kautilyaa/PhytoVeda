"""Loss functions for multi-task learning.

Total loss:
    L_total(t) = w_species(t) * L_species + w_disease(t) * L_disease + lambda * L_reg

- L_species: Cross-Entropy loss for species identification
- L_disease: Focal loss for disease classification (handles class imbalance)
- w_species(t), w_disease(t): Dynamic weights updated per epoch
- L_reg: Orthogonal or entropy regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in pathology classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified examples (healthy leaves) so the model
    focuses on hard, rare disease classes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DynamicTaskWeighting(nn.Module):
    """Dynamic task weighting for multi-task loss balancing.

    Updates w_species(t) and w_disease(t) per epoch based on loss ratios
    to prevent negative transfer and catastrophic forgetting.
    """

    def __init__(self, num_tasks: int = 2, temperature: float = 2.0) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses: torch.Tensor | None = None

    def forward(self, losses: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute dynamically weighted task losses.

        Uses Dynamic Weight Average (DWA): weights are computed from the
        ratio of current to previous epoch losses, normalized via softmax.
        """
        if self.prev_losses is None:
            # First epoch: equal weights
            self.prev_losses = torch.stack([loss.detach() for loss in losses])
            return losses

        current = torch.stack([loss.detach() for loss in losses])
        ratios = current / (self.prev_losses + 1e-8)
        weights = F.softmax(ratios / self.temperature, dim=0) * self.num_tasks

        self.prev_losses = current

        return [w * loss for w, loss in zip(weights, losses, strict=True)]


class CMTLLoss(nn.Module):
    """Combined Multi-Task Learning loss.

    L_total(t) = w_species(t) * L_species + w_disease(t) * L_disease + lambda * L_reg
    """

    def __init__(
        self,
        num_species: int,
        num_pathologies: int = 8,
        focal_gamma: float = 2.0,
        focal_alpha: torch.Tensor | None = None,
        reg_lambda: float = 0.01,
        weighting_temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.species_loss = nn.CrossEntropyLoss()
        self.disease_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dynamic_weighting = DynamicTaskWeighting(
            num_tasks=2, temperature=weighting_temperature
        )
        self.reg_lambda = reg_lambda

    def forward(
        self,
        species_logits: torch.Tensor,
        disease_logits: torch.Tensor,
        species_targets: torch.Tensor,
        disease_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total CMTL loss with dynamic weighting.

        Returns:
            Tuple of (total_loss, loss_dict with individual components).
        """
        l_species = self.species_loss(species_logits, species_targets)
        l_disease = self.disease_loss(disease_logits, disease_targets)

        # Dynamic task weighting
        weighted = self.dynamic_weighting([l_species, l_disease])
        w_species_loss, w_disease_loss = weighted

        # Entropy regularization on species predictions for robustness
        species_probs = F.softmax(species_logits, dim=1)
        l_reg = -(species_probs * torch.log(species_probs + 1e-8)).sum(dim=1).mean()

        total = w_species_loss + w_disease_loss + self.reg_lambda * l_reg

        loss_dict = {
            "total": total.item(),
            "species": l_species.item(),
            "disease": l_disease.item(),
            "reg": l_reg.item(),
            "w_species": (w_species_loss / (l_species + 1e-8)).item(),
            "w_disease": (w_disease_loss / (l_disease + 1e-8)).item(),
        }

        return total, loss_dict

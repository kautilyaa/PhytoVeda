"""Classification heads for the dual-task CMTL architecture.

Species Head: MLP + Softmax, trained with Cross-Entropy loss.
Pathology Head: MLP, trained with Focal Loss for class imbalance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpeciesHead(nn.Module):
    """Species classification head.

    MLP with softmax outputting probability distribution over species classes.
    Trained with standard cross-entropy loss.
    """

    def __init__(
        self,
        in_features: int,
        num_species: int,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_species),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns species logits, shape (B, num_species)."""
        return self.head(x)


class PathologyHead(nn.Module):
    """Pathology classification head.

    MLP outputting disease/health status logits.
    Trained with focal loss to handle severe class imbalance
    (healthy leaves vastly outnumber diseased).
    """

    def __init__(
        self,
        in_features: int,
        num_pathologies: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pathologies),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pathology logits, shape (B, num_pathologies)."""
        return self.head(x)

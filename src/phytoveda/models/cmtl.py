"""Conditional Multi-Task Learning model combining backbone + dual heads.

Architecture:
    Input (B, 3, 512, 512)
        -> HierViT Backbone (shared feature extraction)
        -> Species Head (MLP + softmax, cross-entropy)
        -> Pathology Head (MLP, focal loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from phytoveda.models.backbone import HierViTBackbone
from phytoveda.models.heads import PathologyHead, SpeciesHead


class HierViTCMTL(nn.Module):
    """Hierarchical Vision Transformer with Conditional Multi-Task Learning.

    Dual-head architecture with a shared HierViT backbone (DINOv2/ViT-Huge)
    for simultaneous species identification and pathology diagnosis.
    """

    def __init__(
        self,
        num_species: int,
        num_pathologies: int = 8,
        backbone_name: str = "vit_huge_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        image_size: int = 512,
        species_hidden_dim: int = 1024,
        pathology_hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.backbone = HierViTBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            image_size=image_size,
        )

        feature_dim = self.backbone.feature_dim

        self.species_head = SpeciesHead(
            in_features=feature_dim,
            num_species=num_species,
            hidden_dim=species_hidden_dim,
            dropout=dropout,
        )

        self.pathology_head = PathologyHead(
            in_features=feature_dim,
            num_pathologies=num_pathologies,
            hidden_dim=pathology_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through shared backbone and both classification heads.

        Args:
            x: Batch of images, shape (B, 3, 512, 512).

        Returns:
            Tuple of (species_logits, pathology_logits).
        """
        features = self.backbone(x)
        species_logits = self.species_head(features)
        pathology_logits = self.pathology_head(features)
        return species_logits, pathology_logits

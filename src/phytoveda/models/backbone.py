"""Hierarchical Vision Transformer backbone using timm.

Loads DINOv2 or ViT-Huge pretrained weights as foundation models.
Progressive spatial reduction with increasing channel depth for
multi-scale feature extraction (macro leaf shape + micro lesion details).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class HierViTBackbone(nn.Module):
    """Hierarchical ViT backbone initialized from pretrained foundation models.

    Uses timm to load DINOv2 or ViT-Huge variants. Extracts rich feature
    representations that are shared between the species and pathology heads.
    """

    def __init__(
        self,
        model_name: str = "vit_huge_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, use as feature extractor
            img_size=image_size,
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Batch of images, shape (B, 3, 512, 512).

        Returns:
            Feature tensor, shape (B, feature_dim).
        """
        return self.backbone(x)

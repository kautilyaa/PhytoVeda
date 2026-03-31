"""Basic tests for model architecture."""

import torch

from phytoveda.models.losses import CMTLLoss, FocalLoss


def test_focal_loss_shape() -> None:
    """Focal loss should return a scalar."""
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 8, (4,))
    loss = loss_fn(logits, targets)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_focal_loss_with_alpha() -> None:
    """Focal loss with per-class weights."""
    alpha = torch.ones(8) / 8
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 8, (4,))
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0


def test_cmtl_loss() -> None:
    """CMTL loss should return total loss and component dict."""
    criterion = CMTLLoss(num_species=10, num_pathologies=8)
    species_logits = torch.randn(4, 10)
    disease_logits = torch.randn(4, 8)
    species_targets = torch.randint(0, 10, (4,))
    disease_targets = torch.randint(0, 8, (4,))

    total_loss, loss_dict = criterion(
        species_logits, disease_logits, species_targets, disease_targets
    )

    assert total_loss.shape == ()
    assert "total" in loss_dict
    assert "species" in loss_dict
    assert "disease" in loss_dict

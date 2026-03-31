"""Tests for active learning uncertainty sampling."""

import torch

from phytoveda.active_learning.uncertainty import UncertaintySampler


def test_confident_prediction_not_uncertain() -> None:
    """A highly confident prediction should not be flagged."""
    sampler = UncertaintySampler()
    # One dominant class
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    scores = sampler.score(logits)
    assert len(scores) == 1
    assert not scores[0].is_uncertain


def test_uniform_prediction_is_uncertain() -> None:
    """Equal probabilities across all classes = maximum uncertainty."""
    sampler = UncertaintySampler()
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    scores = sampler.score(logits)
    assert scores[0].is_uncertain


def test_close_top_two_is_uncertain() -> None:
    """Small margin between top-2 predictions = margin uncertainty."""
    sampler = UncertaintySampler(margin_threshold=0.1)
    # Top two are very close
    logits = torch.tensor([[5.0, 4.99, 0.0, 0.0]])
    scores = sampler.score(logits)
    assert scores[0].margin < 0.1


def test_batch_scoring() -> None:
    """Should handle batch of multiple predictions."""
    sampler = UncertaintySampler()
    logits = torch.randn(8, 10)
    scores = sampler.score(logits)
    assert len(scores) == 8

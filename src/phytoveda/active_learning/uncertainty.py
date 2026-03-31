"""Uncertainty sampling strategies for active learning.

Three strategies:
    - Least Confidence: max(softmax_probs) < threshold
    - Margin Sampling: prob_1st - prob_2nd < margin_threshold
    - Entropy: high entropy indicates total uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class UncertaintyScore:
    """Uncertainty assessment for a single prediction."""

    least_confidence: float
    margin: float
    entropy: float
    combined: float
    is_uncertain: bool


class UncertaintySampler:
    """Compute uncertainty scores and flag edge-case predictions."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        margin_threshold: float = 0.1,
        entropy_threshold: float = 2.0,
        combined_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
        self.combined_weights = combined_weights

    def score(self, logits: torch.Tensor) -> list[UncertaintyScore]:
        """Compute uncertainty scores for a batch of predictions.

        Args:
            logits: Raw model output, shape (B, num_classes).

        Returns:
            List of UncertaintyScore for each sample in the batch.
        """
        probs = F.softmax(logits, dim=1)
        sorted_probs, _ = probs.sort(dim=1, descending=True)

        # Least confidence: 1 - max(prob)
        least_conf = 1.0 - sorted_probs[:, 0]

        # Margin: difference between top-2 predictions
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        # Entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        scores = []
        for i in range(logits.size(0)):
            lc = least_conf[i].item()
            mg = margin[i].item()
            ent = entropy[i].item()

            w = self.combined_weights
            combined = w[0] * lc + w[1] * (1.0 - mg) + w[2] * ent

            is_uncertain = (
                lc > (1.0 - self.confidence_threshold)
                or mg < self.margin_threshold
                or ent > self.entropy_threshold
            )

            scores.append(UncertaintyScore(
                least_confidence=lc,
                margin=mg,
                entropy=ent,
                combined=combined,
                is_uncertain=is_uncertain,
            ))

        return scores

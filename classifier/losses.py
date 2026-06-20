"""
Imbalance-aware losses for the factorized classifier.

class_balanced_weights uses the "effective number of samples" reweighting
(Cui et al., CVPR 2019): weight ~ (1 - beta) / (1 - beta**n). Unlike raw
inverse frequency it doesn't explode for a 1-2 sample class, which is exactly
the regime in this dataset.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(counts: list[int], beta: float) -> torch.Tensor:
    counts_t = torch.tensor(counts, dtype=torch.double)
    effective = 1.0 - torch.pow(beta, counts_t)
    weights = (1.0 - beta) / torch.clamp(effective, min=1e-12)
    weights = weights / weights.sum() * len(counts)
    return weights.float()


def counts_for(labels: list[str], vocab: list[str]) -> list[int]:
    freq = Counter(labels)
    return [freq.get(name, 0) for name in vocab]


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(kind: str, weight: torch.Tensor | None, gamma: float,
               label_smoothing: float = 0.0) -> nn.Module:
    if kind in ("ce", "class_balanced"):
        # class_balanced just means CE with class_balanced_weights passed in.
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if kind == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    raise ValueError(f"Unknown loss kind {kind!r}")

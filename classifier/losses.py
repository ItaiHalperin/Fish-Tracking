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


def _circ(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def angular_cost_matrix(class_names: list[str],
                        angle_map: dict[str, tuple[float, float]],
                        heading_weight: float = 1.0,
                        roll_weight: float = 1.0) -> torch.Tensor:
    """[K, K] cost in degrees of predicting class j when the truth is class i.

    Distance is measured on the circle in both orientation axes, so a 45°
    neighbour costs a quarter of what the 180° opposite costs. This is the whole
    mechanism by which the loss learns that some mistakes are worse than others.
    """
    angs = [angle_map[c] for c in class_names]
    return torch.tensor(
        [[heading_weight * _circ(hi, hj) + roll_weight * _circ(ri, rj)
          for hj, rj in angs] for hi, ri in angs], dtype=torch.float32)


def roll_cost_matrix(roll_degrees: list[float]) -> torch.Tensor:
    """[K, K] circular distance between roll classes — belly_down/flank is 90°,
    belly_down/belly_up is 180°."""
    return torch.tensor([[_circ(a, b) for b in roll_degrees] for a in roll_degrees],
                        dtype=torch.float32)


class AngularSoftTargetLoss(nn.Module):
    """Cross-entropy against a distance-aware soft target instead of a one-hot.

    The target for true class i is softmax(-cost[i] / tau), so probability mass
    leaks to orientations near the truth and a near-miss is penalized less than a
    far one. Labels stay plain classes; `cost` carries all the geometry.

    `tau` is in degrees and is the only new knob: as tau -> 0 the soft target
    collapses to the one-hot and this reduces exactly to weighted cross-entropy,
    which makes the comparison against the existing runs a clean ablation.
    """

    def __init__(self, cost: torch.Tensor, tau: float,
                 weight: torch.Tensor | None = None):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0 (use plain CE for tau = 0)")
        self.register_buffer("soft_targets", F.softmax(-cost / tau, dim=1))
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_sample = -(self.soft_targets[target] * F.log_softmax(logits, dim=1)).sum(1)
        if self.weight is None:
            return per_sample.mean()
        w = self.weight[target]
        return (per_sample * w).sum() / w.sum().clamp(min=1e-12)


def build_loss(kind: str, weight: torch.Tensor | None, gamma: float,
               label_smoothing: float = 0.0) -> nn.Module:
    if kind in ("ce", "class_balanced"):
        # class_balanced just means CE with class_balanced_weights passed in.
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if kind == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    raise ValueError(f"Unknown loss kind {kind!r}")

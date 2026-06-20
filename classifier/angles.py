"""
Angle math for the orientation-regression model (Option 3).

Orientation is predicted as continuous angles. To handle the 360° wrap-around
(359° and 1° are 2° apart, not 358°), each angle is encoded as a (cos, sin)
unit vector and the loss is angular, not L2 on the raw degree value.

This module is independent of the class->angle mapping (see configs/angle_map.yaml,
which needs domain definition): it only provides the encode/decode/loss/metric.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml


def load_angle_map(path: str | Path) -> dict[str, tuple[float, float]]:
    """class -> (heading_deg, roll_deg) from configs/angle_map.yaml."""
    raw = yaml.safe_load(Path(path).read_text())
    return {cls: (float(v["heading"]), float(v["roll"])) for cls, v in raw.items()}


ROLL_NAMES = {0: "belly_down", 90: "right_flank", 180: "belly_up", 270: "left_flank"}


def roll_name(deg: float) -> str:
    return ROLL_NAMES.get(int(round(deg)) % 360, f"roll_{int(round(deg))}")


def roll_vocab(angle_map: dict[str, tuple[float, float]]) -> list[float]:
    """Sorted distinct roll values present in the map (the roll classes)."""
    return sorted({r for _, r in angle_map.values()})


def snap_to_class(heading_deg: float, roll_deg: float,
                  angle_map: dict[str, tuple[float, float]]) -> tuple[str, float]:
    """Nearest class to a predicted (heading, roll), plus a 0..1 closeness score
    (1 = exact, falls off with summed angular distance). Used to report a
    discretized accuracy comparable to the classifier."""
    best, best_dist = None, 1e9
    for cls, (h, r) in angle_map.items():
        dh = min((heading_deg - h) % 360, (h - heading_deg) % 360)
        dr = min((roll_deg - r) % 360, (r - roll_deg) % 360)
        dist = dh + dr
        if dist < best_dist:
            best, best_dist = cls, dist
    # 360 = worst case per axis; map summed distance (0..720) to a 1..0 score.
    return best, max(0.0, 1.0 - best_dist / 360.0)


def deg_to_vec(deg: torch.Tensor) -> torch.Tensor:
    """[..., 1] degrees -> [..., 2] (cos, sin) unit vectors."""
    rad = torch.deg2rad(deg)
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)


def vec_to_deg(vec: torch.Tensor) -> torch.Tensor:
    """[..., 2] (cos, sin) -> [...] degrees in [0, 360)."""
    deg = torch.rad2deg(torch.atan2(vec[..., 1], vec[..., 0]))
    return deg % 360.0


def angular_loss(pred_vec: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """1 - cos(error): 0 when aligned, 2 when opposite. pred_vec is the raw
    2-d head output (normalized here), target_deg is ground-truth degrees."""
    pred = torch.nn.functional.normalize(pred_vec, dim=-1)
    target = deg_to_vec(target_deg)
    cos_err = (pred * target).sum(-1)
    return (1.0 - cos_err).mean()


def angular_error_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """Smallest absolute difference on the circle, in degrees [0, 180]."""
    diff = (pred_deg - target_deg).abs() % 360.0
    return torch.minimum(diff, 360.0 - diff)

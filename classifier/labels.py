"""
Composite label algebra.

A composite class is `<pose>_facing_<direction>`, e.g. `diag_up_left_facing_down`.
The factorized model predicts pose and direction separately, so we need to split
a composite into its two factors and recombine predictions back into a composite.
"""

from __future__ import annotations

_FACING_SEP = "_facing_"


def parse_composite(name: str) -> tuple[str, str]:
    """`diag_up_left_facing_down` -> (`diag_up_left`, `down`)."""
    if _FACING_SEP not in name:
        raise ValueError(f"{name!r} is not a composite <pose>_facing_<dir> class")
    pose, facing = name.split(_FACING_SEP, 1)
    return pose, facing


def compose(pose: str, facing: str) -> str:
    return f"{pose}{_FACING_SEP}{facing}"


def _swap_lr(token: str) -> str:
    if token == "left":
        return "right"
    if token == "right":
        return "left"
    return token


def flip_pose_lr(pose: str) -> str:
    """Horizontal mirror swaps any left/right token inside the pose."""
    return "_".join(_swap_lr(t) for t in pose.split("_"))


def flip_facing_lr(facing: str) -> str:
    return _swap_lr(facing)


def flip_composite_lr(name: str) -> str:
    pose, facing = parse_composite(name)
    return compose(flip_pose_lr(pose), flip_facing_lr(facing))

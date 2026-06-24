"""Coerce a crop into a PIL RGB image, so predict() accepts file paths, PIL
images, or in-memory arrays (cv2 frames are BGR, so those get flipped to RGB to
match the training distribution)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    import numpy as np
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # cv2 BGR -> RGB
    return Image.fromarray(arr).convert("RGB")

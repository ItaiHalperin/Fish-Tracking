"""
Resolve a weights path (typically from core.ml_storage) into a ready-to-use
Classifier, so the pipeline is model-agnostic.

A trained run from this package looks like:
    <run>/config.yaml
    <run>/weights/best.pt
We detect the sibling config.yaml, read which model_type it is, and load the
matching Classifier. If there is no config.yaml (e.g. an imported YOLO
`default.pt`), we fall back to a thin YOLO adapter exposing the same interface —
so legacy classifier weights keep working through the same call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import load_config
from .models import build_model
from .models.base import Classifier


def find_run_config(weights_path: Path) -> Path | None:
    """Locate the run's config.yaml given a weights file. Handles the YOLO-native
    layout (<run>/weights/best.pt -> <run>/config.yaml) and a flat layout
    (<run>/best.pt -> <run>/config.yaml)."""
    weights_path = weights_path.resolve()
    for candidate in (weights_path.parents[1] / "config.yaml",
                      weights_path.parent / "config.yaml"):
        if candidate.exists():
            return candidate
    return None


def _set_device(model: Classifier, device: str | None) -> None:
    if not device or not hasattr(model, "_net") or getattr(model, "_net") is None:
        return
    from .models.multihead import _resolve_device
    dev = _resolve_device(device)
    model._device = dev          # type: ignore[attr-defined]
    model._net.to(dev)           # type: ignore[attr-defined]


class _YoloAdapter(Classifier):
    """Wraps an ultralytics YOLO classifier in the Classifier interface."""

    def __init__(self, weights_path: Path, device: str | None = None):
        from ultralytics import YOLO
        self._model = YOLO(str(weights_path))
        self._device = device

    @property
    def class_names(self) -> list[str]:
        return list(self._model.names.values())

    def train(self, config, output_dir):
        raise NotImplementedError("YOLO adapter is inference-only.")

    def load(self, weights_path):
        from ultralytics import YOLO
        self._model = YOLO(str(weights_path))

    def predict(self, image_paths: Iterable[Path | str]) -> list[tuple[str, float]]:
        res = self._model.predict(source=[str(p) for p in image_paths],
                                  device=self._device, verbose=False)
        names = self._model.names
        return [(names[int(r.probs.top1)], float(r.probs.top1conf)) for r in res]


def load_classifier(weights_path: str | Path, device: str | None = None) -> Classifier:
    """Load a Classifier from a weights file. Uses the run's config.yaml to pick
    the model type; falls back to a YOLO adapter if no config is found."""
    weights_path = Path(weights_path)
    cfg_path = find_run_config(weights_path)
    if cfg_path is not None:
        cfg = load_config(cfg_path)
        model = build_model(cfg.model_type)
        model.load(weights_path)
        _set_device(model, device)
        return model
    return _YoloAdapter(weights_path, device)

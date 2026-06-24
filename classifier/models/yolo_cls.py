"""
YOLO11-cls fine-tuning backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from .base import Classifier


class YoloClsClassifier(Classifier):
    def __init__(self):
        self._model: YOLO | None = None

    @property
    def class_names(self) -> list[str]:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded.")
        return list(self._model.names.values())

    def train(self, config, output_dir: Path) -> Path:
        self._model = YOLO(config.pretrained)
        # ultralytics' `project`/`name` controls where runs land. We point it at
        # output_dir so all artifacts go in one place per-config.
        self._model.train(
            data=config.data_dir,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            patience=config.patience,
            seed=config.seed,
            lr0=config.lr0,
            # absolute path — ultralytics treats a relative `project` as relative
            # to its own runs/ dir, which would scatter outputs outside results/.
            project=str(output_dir.parent.resolve()),
            name=output_dir.name,
            exist_ok=True,
            # Augmentation knobs
            degrees=config.aug.degrees,
            translate=config.aug.translate,
            scale=config.aug.scale,
            hsv_h=config.aug.hsv_h,
            hsv_s=config.aug.hsv_s,
            hsv_v=config.aug.hsv_v,
            fliplr=config.aug.fliplr,
            flipud=config.aug.flipud,
            mosaic=config.aug.mosaic,
            erasing=config.aug.erasing,
        )
        weights = output_dir / "weights" / "best.pt"
        if not weights.exists():
            raise RuntimeError(f"Training finished but {weights} is missing.")
        return weights

    def load(self, weights_path: Path) -> None:
        self._model = YOLO(str(weights_path))

    def predict(self, image_paths: Iterable[Path | str]) -> list[tuple[str, float]]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        paths = [str(p) for p in image_paths]
        results = self._model.predict(source=paths, verbose=False)
        names = self._model.names
        out = []
        for r in results:
            idx = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            out.append((names[idx], conf))
        return out

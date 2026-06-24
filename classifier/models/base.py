"""
Abstract base for a fish-position classifier.

Concrete models live alongside this file. Each is registered in
`classifier/models/__init__.py` under a string key that `TrainConfig.model_type`
maps to.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class Classifier(ABC):
    """Minimal interface every classifier must satisfy."""

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Ordered class names; index matches whatever predict() returns."""

    @abstractmethod
    def train(self, config, output_dir: Path) -> Path:
        """
        Train using `config` (a TrainConfig). Drop all artifacts under
        `output_dir`. Return the path to the trained weights file.
        """

    @abstractmethod
    def load(self, weights_path: Path) -> None:
        """Load trained weights into the model."""

    @abstractmethod
    def predict(self, image_paths: Iterable[Path | str]) -> list[tuple[str, float]]:
        """
        For each image, return (predicted_class_name, top1_confidence).
        Confidence is a probability in [0, 1].
        """

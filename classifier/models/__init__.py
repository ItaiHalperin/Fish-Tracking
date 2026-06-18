"""
Model registry. Add new model types here.

`TrainConfig.model_type` is a string lookup against MODEL_REGISTRY.
"""

from .base import Classifier
from .yolo_cls import YoloClsClassifier

MODEL_REGISTRY: dict[str, type[Classifier]] = {
    "yolo_cls": YoloClsClassifier,
    # "logreg_pixels": LogRegPixelsClassifier,  # Phase 4
}


def build_model(model_type: str) -> Classifier:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type]()

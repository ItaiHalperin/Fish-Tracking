"""Fish-position classification package.

`load_classifier` resolves a weights path (e.g. from core.ml_storage) into a
ready Classifier, so the analysis pipeline can stay model-agnostic.
"""

from .loader import load_classifier

__all__ = ["load_classifier"]

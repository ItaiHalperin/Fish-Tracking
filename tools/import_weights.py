#!/usr/bin/env python3
"""
Import pre-trained weights into the MLStorage model registry as the default
weights for a model.

Default weights are stored at:
    ml_storage/model_registry/detection/<model_name>/default.pt
    ml_storage/model_registry/classifier/<model_name>/default.pt

This is intentionally separate from trained runs (run_<timestamp>/).
Use this when you have weights from an external source — downloaded, shared
by a colleague — and want track.py / classify_video.py to pick them up
via --use-storage without having trained locally.

Examples
--------
# Register a detection model
python tools/import_weights.py \\
    --type detection \\
    --weights /path/to/best.pt

# Register a classifier
python tools/import_weights.py \\
    --type classifier \\
    --weights /path/to/best.pt \\
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from core.ml_storage import MLStorage

DEFAULT_DETECTION_MODEL  = "goldfish_yolo"
DEFAULT_CLASSIFIER_MODEL = "fish_position_classifier"


def main():
    parser = argparse.ArgumentParser(
        description="Import external weights into the MLStorage model registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type", required=True, choices=["detection", "classifier"],
        help="Which registry to import into: 'detection' or 'classifier'",
    )
    parser.add_argument(
        "--weights", required=True, type=str,
        help="Path to the weights file to import (e.g. best.pt)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help=(
            f"Model name in the registry "
            f"(default: '{DEFAULT_DETECTION_MODEL}' for detection, "
            f"'{DEFAULT_CLASSIFIER_MODEL}' for classifier)"
        ),
    )
    parser.add_argument(
        "--storage-root", type=str, default="ml_storage",
        help="Root directory for MLStorage (default: ml_storage)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}")
        sys.exit(1)

    # Resolve model name default based on type
    if args.model_name:
        model_name = args.model_name
    elif args.type == "detection":
        model_name = DEFAULT_DETECTION_MODEL
    else:
        model_name = DEFAULT_CLASSIFIER_MODEL

    storage = MLStorage(args.storage_root)
    registry = storage.detection_models if args.type == "detection" else storage.classifier_models

    dst = registry.set_default_weights(model_name=model_name, weights_path=weights_path)

    print(f"\nDone! Default weights set at:\n  {dst}")
    print(f"\nYou can now run:")
    if args.type == "detection":
        print(f"  python detection/track.py --video <video> --output <out> --use-storage")
    else:
        print(f"  python classification/classify_video.py --crops-dir <dir> --output <out> --use-storage")


if __name__ == "__main__":
    main()

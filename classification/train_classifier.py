#!/usr/bin/env python3
"""
Train a 6-class fish position classifier on folder-labeled crops.

Expects this layout (ultralytics ImageFolder convention):

    labels/
      train/
        1_head_down/*.jpg
        2_head_up_diag/*.jpg
        ...
      val/
        1_head_down/*.jpg
        ...

Training runs are saved directly into the MLStorage classifier registry:
    ml_storage/model_registry/classifier/<model_name>/run_<timestamp>/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from ultralytics import YOLO
from core.ml_storage import MLStorage

DEFAULT_MODEL_NAME = "fish_position_classifier"


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO11 fish position classifier.")
    parser.add_argument("--data", default="labels",
                        help="Root dir containing train/ and val/ (default: labels)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    # ML Storage options
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model name in the classifier registry (default: {DEFAULT_MODEL_NAME})")
    args = parser.parse_args()

    base_weights = "yolo11n-cls.pt"
    model = YOLO(base_weights)

    # Prepare the run directory inside the classifier model registry
    storage = MLStorage(args.storage_root)
    project, run_name = storage.classifier_models.prepare_run(args.model_name)
    print(f"Training output: {project}/{run_name}/")

    print("Starting classifier training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project,
        name=run_name,
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
    )
    print("Training completed successfully!")

    # Finalize the run with metadata
    save_dir = Path(results.save_dir)
    storage.classifier_models.finalize_run(
        model_name=args.model_name,
        run_dir=save_dir,
        metadata={
            "base_model": base_weights,
            "data": args.data,
            "epochs_requested": args.epochs,
            "imgsz": args.imgsz,
        },
    )


if __name__ == "__main__":
    main()

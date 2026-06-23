#!/usr/bin/env python3
"""
YOLOv11 Training Script for FishTracking.

Training runs are saved directly into the MLStorage model registry.
All YOLO artefacts (weights, metrics, plots) live in
``ml_storage/model_registry/<model_name>/run_<timestamp>/``.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from core.ml_storage import MLStorage

VALID_MODELS = ["n", "s", "m", "l", "x"]
DEFAULT_MODEL_NAME = "goldfish_yolo"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on the fish tracking dataset.")
    parser.add_argument("--model", choices=VALID_MODELS, default="n",
                        help="Model size: n(ano), s(mall), m(edium), l(arge), x (default: n)")
    parser.add_argument("--epochs", default="200",
                        help="Number of epochs (default: 200)")
    # ML Storage options
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model name in the registry (default: {DEFAULT_MODEL_NAME})")
    args = parser.parse_args()

    base_weights = f"yolo11{args.model}.pt"
    print(f"Initializing YOLOv11-{args.model} from {base_weights}...")
    model = YOLO(base_weights)

    # Prepare the run directory inside the detection model registry
    storage = MLStorage(args.storage_root)
    project, run_name = storage.detection_models.prepare_run(args.model_name)
    print(f"Training output: {project}/{run_name}/")

    # Start the training loop
    print("Starting training...")
    try:
        results = model.train(
            data="dataset.yaml",
            epochs=int(args.epochs),
            patience=50,
            imgsz=640,
            batch=16,
            device='mps',

            # Save directly into the model registry
            project=project,
            name=run_name,
            
            # --- Geometric Augmentations ---
            degrees=10.0,    
            translate=0.1,   
            scale=0.5,       
            fliplr=0.5,      
            flipud=0.2,      
            mosaic=1.0,      
            mixup=0.1,       
            erasing=0.3,      # Randomly erase 30% of patches, forces learning from partial views

            # --- Color Augmentations ---
            hsv_h=0.015,      # Hue shift
            hsv_s=0.7,        # Saturation shift
            hsv_v=0.4         # Brightness shift
        )
        print("Training completed successfully!")

        # Finalize the run with metadata
        save_dir = Path(results.save_dir)
        storage.detection_models.finalize_run(
            model_name=args.model_name,
            run_dir=save_dir,
            metadata={
                "base_model": base_weights,
                "model_size": args.model,
                "epochs_requested": int(args.epochs),
            },
        )

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

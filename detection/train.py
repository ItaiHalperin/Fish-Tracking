#!/usr/bin/env python3
"""
YOLO Training Script for FishTracking.

Supports YOLO v5, v11, and v26 in sizes nano/small/medium/large/xlarge.
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
from core.device import get_device

# Map full size names to single-letter codes (short forms also accepted directly)
SIZE_ALIASES = {
    "nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x",
    "n": "n", "s": "s", "m": "m", "l": "l", "x": "x",
}
VALID_SIZES = list(SIZE_ALIASES.keys())
VALID_VERSIONS = ["5", "11", "26"]
DEFAULT_MODEL_NAME = "goldfish_yolo"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on the fish tracking dataset.")
    parser.add_argument("--model", choices=VALID_SIZES, default="n",
                        help="Model size: nano/n, small/s, medium/m, large/l, xlarge/x (default: n)")
    parser.add_argument("--version", choices=VALID_VERSIONS, default="11",
                        help=f"YOLO version to use ({', '.join(VALID_VERSIONS)}) (default: 11)")
    parser.add_argument("--epochs", default="200",
                        help="Number of epochs (default: 200)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset.yaml file (defaults to latest in MLStorage)")
    # ML Storage options
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model name in the registry (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to custom pretrained weights (e.g., best.pt). Overrides standard base weights.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate (e.g., 0.001 for fine-tuning). If not set, YOLO auto-determines.")
    args = parser.parse_args()

    # Resolve size alias to single-letter code
    size = SIZE_ALIASES[args.model]

    # Determine base weights string based on YOLO version naming conventions
    weight_patterns = {
        "5":  f"yolov5{size}u.pt",   # Ultralytics updated v5 models have 'u' suffix
        "11": f"yolo11{size}.pt",
        "26": f"yolo26{size}.pt",
    }
    base_weights = args.weights if args.weights else weight_patterns[args.version]
        
    print(f"Initializing YOLOv{args.version}-{size} from {base_weights}...")
    model = YOLO(base_weights)

    # Prepare the run directory inside the detection model registry
    storage = MLStorage(args.storage_root)
    
    # Resolve Dataset
    dataset_path = args.dataset
    if not dataset_path:
        latest_ds_dir = storage.datasets.latest_version()
        if latest_ds_dir:
            dataset_path = str(latest_ds_dir / "dataset.yaml")
        else:
            print("Error: No datasets found in MLStorage and --dataset not provided.", file=sys.stderr)
            sys.exit(1)
            
    # Automatically tag the model name with the version if it's using the default
    model_name = args.model_name
    if model_name == DEFAULT_MODEL_NAME and args.version != "11":
        model_name = f"goldfish_yolov{args.version}"
        
    project, run_name = storage.detection_models.prepare_run(model_name)
    print(f"Training output: {project}/{run_name}/")

    # Build kwargs for train
    train_kwargs = dict(
        data=dataset_path,
        epochs=int(args.epochs),
        patience=50,
        imgsz=640,
        batch=16,
        device=get_device(),

        # Save directly into the model registry
        project=project,
        name=run_name,

        # --- Geometric Augmentations ---
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.2,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        erasing=0.3,

        # --- Color Augmentations ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )

    if args.lr is not None:
        train_kwargs["lr0"] = args.lr
        train_kwargs["lrf"] = 0.01 # Final learning rate ratio (lr0 * lrf)

    # Start the training loop
    print("Starting training...")
    try:
        results = model.train(**train_kwargs)
        print("Training completed successfully!")

        # Finalize the run with metadata
        save_dir = Path(results.save_dir)
        storage.detection_models.finalize_run(
            model_name=model_name,
            run_dir=save_dir,
            metadata={
                "base_model": base_weights,
                "yolo_version": args.version,
                "model_size": size,
                "epochs_requested": int(args.epochs),
            },
        )

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

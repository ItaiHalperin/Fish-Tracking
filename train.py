#!/usr/bin/env python3
"""
YOLOv11 Training Script for FishTracking.
"""

import argparse
import sys
from ultralytics import YOLO

VALID_MODELS = ["n", "s", "m", "l", "x"]

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on the fish tracking dataset.")
    parser.add_argument("--model", choices=VALID_MODELS, default="n",
                        help="Model size: n(ano), s(mall), m(edium), l(arge), x (default: n)")
    args = parser.parse_args()

    weights = f"yolo11{args.model}.pt"
    print(f"Initializing YOLOv11-{args.model} from {weights}...")
    model = YOLO(weights)

    # Start the training loop
    print("Starting training...")
    try:
        results = model.train(
            data="dataset.yaml",
            epochs=200,      # Increased epochs for more learning
            patience=50,     # Early stopping if no improvement after 50 epochs
            imgsz=640,       # Image size for training
            batch=16,        # Batch size
            device='mps',    # Force Apple Silicon GPU acceleration
            
            # Removed the `project="runs/detect"` argument so YOLO defaults to its normal
            # 'runs/detect' folder instead of nesting it!
            name="fish_tracking_model",
            
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
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

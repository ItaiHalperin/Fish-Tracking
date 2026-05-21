#!/usr/bin/env python3
"""
YOLOv11 Training Script for FishTracking.
"""

import sys
from ultralytics import YOLO

def main():
    # Load a pretrained YOLO11 Nano model
    print("Initializing YOLOv11 model...")
    model = YOLO("yolo11n.pt")

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
            
            # --- Data Augmentations ---
            degrees=10.0,    
            translate=0.1,   
            scale=0.5,       
            fliplr=0.5,      
            flipud=0.2,      
            mosaic=1.0,      
            mixup=0.1,       
            hsv_h=0.015,     
            hsv_s=0.7,       
            hsv_v=0.4        
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

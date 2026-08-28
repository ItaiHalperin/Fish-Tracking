#!/usr/bin/env python3
"""
Test script to evaluate Faster R-CNN on the test dataset using mAP.
"""

import sys
import yaml
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall

# Add parent directory to path to import core modules if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import dataset and model utilities from train_faster_rcnn.py
from train_faster_rcnn import YoloToFasterRCNNDataset, get_model
from core.device import get_device

def get_latest_weights_path(storage_root, model_name):
    """Finds the best.pt weights from the latest run."""
    model_dir = Path(storage_root) / "model_registry" / "detection" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    # Get all run directories, sort them by name (timestamp)
    runs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        raise FileNotFoundError(f"No runs found for model: {model_name}")
        
    latest_run = runs[-1]
    weights_path = latest_run / "weights" / "best.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    return weights_path

def main():
    parser = argparse.ArgumentParser(description="Test Faster R-CNN model on the test dataset.")
    parser.add_argument("--weights", type=str, default="", help="Path to weights (if empty, uses latest run)")
    parser.add_argument("--storage-root", type=str, default="ml_storage", help="Root directory for MLStorage")
    parser.add_argument("--model-name", type=str, default="goldfish_faster_rcnn", help="Model name in the registry")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    args = parser.parse_args()

    device = torch.device(get_device())
    print(f"Using device: {device}")

    # Determine weights path
    weights_path = args.weights
    if not weights_path:
        weights_path = str(get_latest_weights_path(args.storage_root, args.model_name))
    print(f"Using weights: {weights_path}")

    # The model was trained with 2 classes (0: background, 1: goldfish)
    num_classes = 2

    # Initialize model
    print("Loading model...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load dataset
    print("Loading test dataset...")
    test_dataset = YoloToFasterRCNNDataset("ml_storage/datasets/v1_initial_training_set/dataset.yaml", split="test")
    
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty!")
        return

    # Initialize metrics
    map_metric = MeanAveragePrecision()
    precision_metric = Precision()
    recall_metric = Recall()
    
    print("Evaluating...")
    
    # We iterate over the dataset manually to easily handle supervision Detections conversion
    for i in tqdm(range(len(test_dataset))):
        image_tensor, target = test_dataset[i]
        
        # Ground Truth
        if len(target["boxes"]) > 0:
            target_detections = sv.Detections(
                xyxy=target["boxes"].numpy(),
                class_id=target["labels"].numpy()
            )
        else:
            target_detections = sv.Detections.empty()

        # Prediction
        with torch.no_grad():
            image_tensor_float = image_tensor.to(device).float() / 255.0
            outputs = model([image_tensor_float])[0]
            
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        class_ids = outputs["labels"].cpu().numpy()
        
        # Filter by confidence
        mask = scores >= args.conf
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) > 0:
            pred_detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=class_ids
            )
        else:
            pred_detections = sv.Detections.empty()

        # Update metrics
        map_metric.update(predictions=[pred_detections], targets=[target_detections])
        precision_metric.update(predictions=[pred_detections], targets=[target_detections])
        recall_metric.update(predictions=[pred_detections], targets=[target_detections])

    # Compute and display results
    print("\nComputing metrics...")
    map_result = map_metric.compute()
    precision_result = precision_metric.compute()
    recall_result = recall_metric.compute()
    
    print("\n" + "="*50)
    print("Faster R-CNN Evaluation Results")
    print("="*50)
    print(f"mAP@50:95:      {map_result.map50_95:.4f}")
    print(f"mAP@50:         {map_result.map50:.4f}")
    print(f"mAP@75:         {map_result.map75:.4f}")
    print(f"Precision@50:   {precision_result.precision_at_50:.4f}")
    print(f"Recall@50:      {recall_result.recall_at_50:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Faster R-CNN Video Inference Script.

This script runs the trained PyTorch Faster R-CNN model on a video
and outputs an annotated video with bounding boxes (similar to the YOLO output,
but without the temporal tracking/ByteTrack IDs).
"""

import sys
import argparse
from pathlib import Path

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.device import get_device

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    parser = argparse.ArgumentParser(description="Generate annotated video using Faster R-CNN.")
    parser.add_argument("--video", required=True, type=str, help="Path to input video")
    parser.add_argument("--output", required=True, type=str, help="Path to output video (.mp4)")
    parser.add_argument("--weights", required=True, type=str, help="Path to trained .pt weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes including background (default: 3)")
    args = parser.parse_args()

    device = torch.device(get_device())
    print(f"Using device: {device}")

    print(f"Loading model from {args.weights}...")
    model = get_model(args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video {args.video} ({total_frames} frames)...")
    
    frame_count = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")

            # Convert BGR (OpenCV) to RGB (PyTorch)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = F.to_tensor(rgb_frame).to(device)

            # Inference
            predictions = model([tensor_frame])[0]

            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()

            # Draw boxes
            for box, score, label in zip(boxes, scores, labels):
                if score >= args.conf:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label and score
                    text = f"Class {label-1}: {score:.2f}" # -1 to revert background shift
                    cv2.putText(frame, text, (x1, max(y1 - 10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

    cap.release()
    out.release()
    print(f"Saved annotated video to {args.output}")

if __name__ == "__main__":
    main()

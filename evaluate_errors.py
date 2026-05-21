#!/usr/bin/env python3
"""
Error Analysis Script
Runs the model over the validation (testing) set and compares the model's predicted
bounding boxes to the ground truth annotations. 
If the model hallucinates a fish (False Positive) or misses a fish (False Negative),
it exports the image with the Ground Truth drawn in GREEN and the Prediction in RED.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

def read_yolo_labels(label_path, img_width, img_height):
    """Reads YOLO normalized .txt labels and converts them to absolute pixel coordinates."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # Convert normalized xywh to absolute xyxy
                x1 = int((x_center - w / 2) * img_width)
                y1 = int((y_center - h / 2) * img_height)
                x2 = int((x_center + w / 2) * img_width)
                y2 = int((y_center + h / 2) * img_height)
                boxes.append((class_id, x1, y1, x2, y2))
    return boxes

def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def main():
    # Load the best weights from your most recent training run
    model_path = "runs/detect/fish_tracking_model-3/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Error: Could not find {model_path}. Please check the path.")
        return
        
    model = YOLO(model_path)
    
    test_images_dir = Path("dataset/images/test")
    test_labels_dir = Path("dataset/labels/test")
    output_dir = Path("runs/detect/errors")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating testing images in '{test_images_dir}'...")
    
    error_count = 0
    # Process every image in your testing set
    for img_path in test_images_dir.glob("*.jpg"):
        
        # 1. Run YOLO Prediction
        results = model.predict(source=str(img_path), conf=0.25, verbose=False, device="mps")
        result = results[0]
        
        # 2. Load the actual image for drawing
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]
        
        # 3. Read the real Ground Truth labels you annotated
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = read_yolo_labels(label_path, img_width, img_height)
        
        # 4. Extract Predicted boxes
        pred_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pred_boxes.append((x1, y1, x2, y2))
            
        # 5. Logic to detect if the model made a mistake
        has_error = False
        
        # Check for False Positives (Model drew a box, but it doesn't overlap a GT box)
        for p_box in pred_boxes:
            matched = False
            for _, gx1, gy1, gx2, gy2 in gt_boxes:
                if compute_iou(p_box, (gx1, gy1, gx2, gy2)) > 0.3:
                    matched = True
                    break
            if not matched:
                has_error = True
                break
                
        # Check for False Negatives (GT box exists, but model completely missed it)
        if not has_error:
            for _, gx1, gy1, gx2, gy2 in gt_boxes:
                matched = False
                for p_box in pred_boxes:
                    if compute_iou(p_box, (gx1, gy1, gx2, gy2)) > 0.3:
                        matched = True
                        break
                if not matched:
                    has_error = True
                    break
                    
        # 6. If it made a mistake, export the visual evidence
        if has_error:
            error_count += 1
            
            # Draw Ground Truth in GREEN
            for _, x1, y1, x2, y2 in gt_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "GT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Draw Prediction in RED
            for x1, y1, x2, y2 in pred_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "PRED", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), img)
            
    print(f"\nDone! Found {error_count} images where the model made a mistake.")
    print(f"Check the '{output_dir}' folder to see the errors!")
    print("Green Box = Ground Truth (What it should be)")
    print("Red Box   = Model Prediction (What it guessed)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Faster R-CNN Training Script for FishTracking.

This script utilizes the same labeled data (YOLO format) and MLStorage 
as the YOLO training script, serving as a baseline comparison.
"""

import sys
import yaml
import json
import argparse
from pathlib import Path
from PIL import Image
from functools import lru_cache

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import supervision as sv
from supervision.metrics import MeanAveragePrecision, Precision, Recall

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.ml_storage import MLStorage
from core.device import get_device

# --- Dataset Class ---

class YoloToFasterRCNNDataset(torch.utils.data.Dataset):
    """
    Parses YOLO format datasets (images and .txt labels) and 
    provides them in the format expected by torchvision's Faster R-CNN.
    """
    def __init__(self, yaml_path, split="train", transform=None):
        self.transform = transform
        self.yaml_path = Path(yaml_path)
        with open(self.yaml_path, "r") as f:
            self.data_config = yaml.safe_load(f)
        
        self.root_dir = Path(self.data_config["path"])
        # e.g., dataset/images/train
        self.img_dir = self.root_dir / self.data_config[split]
        # Assuming labels are in dataset/labels/train
        self.label_dir = self.img_dir.parent.parent / "labels" / self.img_dir.name
        
        # Valid image extensions
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_exts]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Find corresponding label file
        label_path = self.label_dir / (img_path.stem + ".txt")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO to absolute and clamp to image dimensions
                    xmin = max(0.0, (x_center - width / 2) * w)
                    ymin = max(0.0, (y_center - height / 2) * h)
                    xmax = min(float(w), (x_center + width / 2) * w)
                    ymax = min(float(h), (y_center + height / 2) * h)
                    
                    # Only append valid boxes
                    if xmax > xmin and ymax > ymin:
                        # We only have one actual class (goldfish, YOLO class 1).
                        # Map it to Faster R-CNN class 1 (class 0 is background).
                        if class_id == 1:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(1)
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))
        else:
            # Handle images with no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))
            
        image_tensor = tv_tensors.Image(image)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transform is not None:
            image_tensor, target = self.transform(image_tensor, target)
            
        boxes = target["boxes"]
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# --- Model Creation ---

def get_model(num_classes):
    # Load a pre-trained MobileNetV3 model which is MUCH faster and lighter than ResNet50
    # Load a pre-trained MobileNetV3 model which is MUCH faster and lighter than ResNet50
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on the fish tracking dataset as a baseline.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size (default: 16)")
    parser.add_argument("--num-workers", default=0, type=int, help="Number of dataloader workers. Try 2 or 4 if CPU is bottlenecking, but be aware of macOS spawn deadlocks. (default: 0)")
    # ML Storage options
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--model-name", type=str, default="goldfish_faster_rcnn",
                        help="Model name in the registry (default: goldfish_faster_rcnn)")
    args = parser.parse_args()

    device = torch.device(get_device())
    print(f"Using device: {device}")

    # Since YOLO class 0 is unused and class 1 is goldfish, we just use 2 classes (bg + goldfish)
    num_classes = 2

    # Define augmentations to mimic YOLO's behavior
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomAffine(degrees=10.0, translate=(0.1, 0.1), scale=(0.5, 1.5)),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
        v2.ToDtype(torch.float32, scale=True),
        v2.SanitizeBoundingBoxes(), # removes boxes that are outside the image after transform
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # Prepare datasets
    print("Loading datasets...")
    dataset_train = YoloToFasterRCNNDataset("dataset.yaml", split="train", transform=train_transform)
    dataset_val = YoloToFasterRCNNDataset("dataset.yaml", split="val", transform=val_transform)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f"Initializing Faster R-CNN with {num_classes} classes...")
    model = get_model(num_classes)
    model.to(device)

    # Optimizer - SGD is standard and much more stable for Faster R-CNN than AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    storage = MLStorage(args.storage_root)
    project, run_name = storage.detection_models.prepare_run(args.model_name)
    run_dir = Path(project) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training output: {run_dir}/")

    # Mixed-precision for faster training (CUDA only; MPS lacks float16 support in roi_align)
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("Starting training...")
    best_map50 = 0.0
    import numpy as np
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader_train):
            print(f"  Training batch {i+1}/{len(data_loader_train)}...")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Skip NaN batches to prevent poisoning the model weights
            if not torch.isfinite(losses):
                print(f"    WARNING: Non-finite loss at batch {i+1}, skipping. Components: {{{', '.join(f'{k}: {v.item():.4f}' for k, v in loss_dict.items())}}}")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            if i == len(data_loader_train) - 1:
                components = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                print(f"    Loss components: {components}")

        lr_scheduler.step()
        
        avg_loss = epoch_loss / len(data_loader_train)
        print(f"Epoch: {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        print("  Evaluating on validation set...")
        model.eval()
        map_metric = MeanAveragePrecision()
        precision_metric = Precision()
        recall_metric = Recall()
        
        for images_val, targets_val in data_loader_val:
            images_val = list(img.to(device) for img in images_val)
            
            with torch.no_grad():
                outputs = model(images_val)
                
            for target_val, output in zip(targets_val, outputs):
                if len(target_val["boxes"]) > 0:
                    target_detections = sv.Detections(
                        xyxy=target_val["boxes"].numpy(),
                        class_id=target_val["labels"].numpy()
                    )
                else:
                    target_detections = sv.Detections.empty()
                    
                # Predictions (filter by 0.05 conf for mAP calculation standard)
                boxes_out = output["boxes"].cpu().numpy()
                scores_out = output["scores"].cpu().numpy()
                class_ids_out = output["labels"].cpu().numpy()
                
                mask = scores_out >= 0.05
                if np.any(mask):
                    pred_detections = sv.Detections(
                        xyxy=boxes_out[mask],
                        confidence=scores_out[mask],
                        class_id=class_ids_out[mask]
                    )
                else:
                    pred_detections = sv.Detections.empty()
                    
                map_metric.update(predictions=[pred_detections], targets=[target_detections])
                precision_metric.update(predictions=[pred_detections], targets=[target_detections])
                recall_metric.update(predictions=[pred_detections], targets=[target_detections])
                
        map_res = map_metric.compute()
        prec_res = precision_metric.compute()
        rec_res = recall_metric.compute()
        current_map50 = map_res.map50
        
        print(f"  val_mAP@50: {current_map50:.4f} | val_mAP@50-95: {map_res.map50_95:.4f} | val_Precision: {prec_res.precision_at_50:.4f} | val_Recall: {rec_res.recall_at_50:.4f}")
        
        torch.save(model.state_dict(), weights_dir / "last.pt")
        
        if current_map50 > best_map50:
            best_map50 = current_map50
            print(f"  --> New best model! Saving best.pt (mAP@50: {best_map50:.4f})")
            torch.save(model.state_dict(), weights_dir / "best.pt")

    print("Training completed successfully!")

    storage.detection_models.finalize_run(
        model_name=args.model_name,
        run_dir=run_dir,
        metadata={
            "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "num_classes": num_classes
        },
    )

if __name__ == "__main__":
    main()

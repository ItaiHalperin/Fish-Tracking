#!/usr/bin/env python3
"""
COCO to YOLO Dataset Ingestion Script.

Reads a ZIP file containing a COCO dataset (with a 'train' folder and '_annotations.coco.json'),
unzips it, extracts the images, converts the bounding boxes to YOLO format, and
sets up the dataset directory structure and dataset.yaml for YOLOv11 training.
"""

import argparse
import json
import random
import shutil
import tempfile
import yaml
import zipfile
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a COCO zip and convert to a YOLOv11 dataset.")
    parser.add_argument("--zip-file", required=True, type=str, help="Path to the .zip file containing the COCO dataset")
    parser.add_argument("--output-dir", default="dataset", type=str, help="Output directory for the YOLO dataset (default: dataset)")
    parser.add_argument("--val-ratio", default=0.15, type=float, help="Fraction of the data to use for validation (default: 0.15)")
    parser.add_argument("--test-ratio", default=0.15, type=float, help="Fraction of the data to use for testing (default: 0.15)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for the train/val/test split (default: 42)")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    zip_path = Path(args.zip_file)
    if not zip_path.exists():
        print(f"Error: Zip file '{zip_path}' not found.")
        return
        
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    # Create the YOLO destination structure
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        
    # Use a temporary directory to unzip the files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Extracting {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
        except zipfile.BadZipFile:
            print("Error: The provided file is not a valid zip archive.")
            return
            
        # Locate the 'train' folder dynamically in case it's nested
        train_dir = None
        for p in temp_path.rglob("train"):
            if p.is_dir():
                train_dir = p
                break
                
        if not train_dir:
            print("Error: Could not find a 'train' folder inside the zip.")
            return
            
        json_file = train_dir / "_annotations.coco.json"
        if not json_file.exists():
            print(f"Error: Annotation file '_annotations.coco.json' not found inside '{train_dir}'.")
            return
            
        print("Parsing COCO JSON...")
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
            
        # Extract categories (e.g., id: 0, name: 'fish')
        # We assign a new 0-indexed YOLO class id for each category
        categories = {}
        for idx, cat in enumerate(coco_data.get("categories", [])):
            categories[cat["id"]] = (idx, cat["name"])
            
        if not categories:
            print("Error: No categories found in the JSON file.")
            return
            
        # Group annotations by image_id for fast lookup
        annotations_by_img = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            annotations_by_img.setdefault(img_id, []).append(ann)
            
        # Get images and shuffle for splitting
        images = coco_data.get("images", [])
        if not images:
            print("Error: No images found in the JSON file.")
            return
            
        random.shuffle(images)
        val_count = int(round(len(images) * args.val_ratio))
        test_count = int(round(len(images) * args.test_ratio))
        
        val_images = images[:val_count]
        test_images = images[val_count:val_count + test_count]
        train_images = images[val_count + test_count:]
        
        def process_split(img_list, split_name):
            count = 0
            for img in img_list:
                img_id = img.get("id")
                file_name = img.get("file_name")
                width = img.get("width")
                height = img.get("height")
                
                src_img_path = train_dir / file_name
                if not src_img_path.exists():
                    print(f"Warning: Image '{file_name}' referenced in JSON not found in zip. Skipping.")
                    continue
                    
                # 1. Copy Image
                dst_img_path = images_dir / split_name / file_name
                shutil.copy2(src_img_path, dst_img_path)
                
                # 2. Write YOLO Label
                label_name = Path(file_name).stem + ".txt"
                label_path = labels_dir / split_name / label_name
                
                with open(label_path, 'w') as lf:
                    anns = annotations_by_img.get(img_id, [])
                    for ann in anns:
                        cat_id = ann["category_id"]
                        if cat_id not in categories:
                            continue
                            
                        yolo_class_id, _ = categories[cat_id]
                        
                        # COCO uses absolute coordinates: [x_min, y_min, width, height]
                        x_min, y_min, w, h = ann["bbox"]
                        x_min, y_min, w, h = float(x_min), float(y_min), float(w), float(h)
                        
                        # YOLO uses normalized coordinates: [x_center, y_center, norm_w, norm_h]
                        img_w, img_h = float(width), float(height)
                        x_center = (x_min + w / 2) / img_w
                        y_center = (y_min + h / 2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        
                        # Clip values strictly between 0 and 1 to prevent YOLO validation errors
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))
                        
                        lf.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                count += 1
            return count
            
        print(f"Converting coordinates and copying files (Val {args.val_ratio*100:.0f}%, Test {args.test_ratio*100:.0f}%)...")
        train_processed = process_split(train_images, "train")
        val_processed = process_split(val_images, "val")
        test_processed = process_split(test_images, "test")
        
        print(f"Successfully processed {train_processed} training, {val_processed} validation, and {test_processed} testing images.")
        
        # Generate the new dataset.yaml
        yaml_path = Path("dataset.yaml")
        names_dict = {idx: name for idx, name in categories.values()}
        
        yaml_content = {
            "path": str(output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names_dict
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        print(f"\nCreated YOLO configuration at: {yaml_path}")
        print(f"Detected classes: {names_dict}")
        print("\nAll done! You can now start training by running:")
        print("  .venv/bin/python train.py")

if __name__ == "__main__":
    main()

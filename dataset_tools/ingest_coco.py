#!/usr/bin/env python3
"""
COCO to YOLO Dataset Ingestion.

This module provides a CocoIngestor class that reads a ZIP file containing a
COCO dataset (with pre-existing 'train', 'valid', and 'test' folders), unzips
it, extracts the images, converts the bounding boxes to YOLO format, and sets
up the dataset directory structure and dataset.yaml for YOLOv11 training while
preserving the original splits.

Can be used as a standalone script (python ingest_coco.py ...) or imported as
a class in other modules. Optionally outputs into versioned MLStorage.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime
import json
import shutil
import tempfile
import yaml
import zipfile
from pathlib import Path

from core.ml_storage import MLStorage


class CocoIngestor:
    """Ingests a COCO-format ZIP dataset and converts it to YOLO format."""

    def __init__(
        self,
        zip_file: str,
        output_dir: str = "dataset",
        storage: MLStorage | None = None,
        version_description: str | None = None,
    ):
        self.zip_path = Path(zip_file)
        self.output_dir = Path(output_dir)
        self.storage = storage
        self.version_description = version_description

    def _resolve_output_dir(self) -> Path:
        """Determine the output directory, optionally creating a versioned dataset."""
        if self.storage is not None:
            desc = self.version_description or f"coco_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version_dir = self.storage.datasets.create_version(desc)
            return version_dir
        return self.output_dir

    @staticmethod
    def _convert_split(
        split_dir: Path,
        images_dir: Path,
        labels_dir: Path,
        yolo_split: str,
        all_categories: dict,
    ) -> int:
        """Convert a single COCO split folder into YOLO images + labels.

        Returns the number of images processed.
        """
        json_file = split_dir / "_annotations.coco.json"
        if not json_file.exists():
            print(f"Warning: Annotation file '_annotations.coco.json' not found inside '{split_dir}'. Skipping.")
            return 0

        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        # Extract categories. Roboflow exports consistent categories across all splits.
        for idx, cat in enumerate(coco_data.get("categories", [])):
            all_categories[cat["id"]] = (idx, cat["name"])

        # Group annotations by image_id
        annotations_by_img = {}
        for ann in coco_data.get("annotations", []):
            annotations_by_img.setdefault(ann["image_id"], []).append(ann)

        images = coco_data.get("images", [])
        count = 0

        for img in images:
            img_id = img.get("id")
            file_name = img.get("file_name")
            width = img.get("width")
            height = img.get("height")

            src_img_path = split_dir / file_name
            if not src_img_path.exists():
                print(f"  Warning: Image '{file_name}' not found. Skipping.")
                continue

            # 1. Copy Image
            dst_img_path = images_dir / yolo_split / file_name
            shutil.copy2(src_img_path, dst_img_path)

            # 2. Write YOLO Label
            label_name = Path(file_name).stem + ".txt"
            label_path = labels_dir / yolo_split / label_name

            with open(label_path, 'w') as lf:
                for ann in annotations_by_img.get(img_id, []):
                    cat_id = ann["category_id"]
                    if cat_id not in all_categories:
                        continue

                    yolo_class_id, _ = all_categories[cat_id]

                    # COCO uses absolute coordinates: [x_min, y_min, width, height]
                    x_min, y_min, w, h = map(float, ann["bbox"])
                    img_w, img_h = float(width), float(height)

                    # YOLO uses normalized coordinates: [x_center, y_center, norm_w, norm_h]
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

    def run(self) -> None:
        """Execute the full COCO → YOLO ingestion pipeline."""
        if not self.zip_path.exists():
            print(f"Error: Zip file '{self.zip_path}' not found.")
            return

        effective_output = self._resolve_output_dir()
        images_dir = effective_output / "images"
        labels_dir = effective_output / "labels"

        # Create the YOLO destination structure
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

        # Use a temporary directory to unzip the files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            print(f"Extracting {self.zip_path.name}...")

            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
            except zipfile.BadZipFile:
                print("Error: The provided file is not a valid zip archive.")
                return

            all_categories = {}

            # Iterate over the expected COCO splits inside the zip
            for coco_split, yolo_split in [("train", "train"), ("valid", "val"), ("test", "test")]:
                # Locate the split folder dynamically in case it's nested
                split_dir = None
                for p in temp_path.rglob(coco_split):
                    if p.is_dir() and (p.parent == temp_path or "unzipped" in p.parent.name.lower() or p.parent.name == self.zip_path.stem):
                        split_dir = p
                        break

                if not split_dir:
                    print(f"Notice: Could not find a '{coco_split}' folder inside the zip. Skipping.")
                    continue

                print(f"\nProcessing '{coco_split}' split...")
                count = self._convert_split(split_dir, images_dir, labels_dir, yolo_split, all_categories)
                print(f"  -> Converted {count} images to {yolo_split}.")

            if not all_categories:
                print("\nError: No categories found. Aborting.")
                return

            # Generate the dataset.yaml inside the output directory
            yaml_path = effective_output / "dataset.yaml"
            names_dict = {idx: name for idx, name in all_categories.values()}

            yaml_content = {
                "path": str(effective_output.absolute()),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a pre-split COCO zip and convert to YOLOv11 dataset.")
    parser.add_argument("--zip-file", required=True, type=str, help="Path to the .zip file containing the COCO dataset")
    parser.add_argument("--output-dir", default="dataset", type=str, help="Output directory for the YOLO dataset (default: dataset)")
    # ML Storage options
    parser.add_argument("--use-storage", action="store_true",
                        help="Enable versioned dataset storage via MLStorage")
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--version-desc", type=str, default=None,
                        help="Description for this dataset version (e.g. 'initial COCO import')")
    return parser.parse_args()


def main():
    args = parse_args()

    storage = None
    if args.use_storage:
        storage = MLStorage(args.storage_root)

    ingestor = CocoIngestor(
        zip_file=args.zip_file,
        output_dir=args.output_dir,
        storage=storage,
        version_description=args.version_desc,
    )
    ingestor.run()


if __name__ == "__main__":
    main()

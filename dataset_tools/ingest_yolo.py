#!/usr/bin/env python3
"""
YOLO Dataset Ingestion & Merging.

Ingests one or more ZIP files containing pre-formatted YOLO datasets
(images + .txt label files), optionally remaps class IDs to a unified
class set, and outputs a single merged YOLO dataset.

Handles common layout variants:
  - Flat:   {train,valid,test}/{images,labels}/
  - Nested: <name>/{train,valid,testb}/{images,labels}/
  - Roboflow & RUOD naming conventions (valid→val, testb→test)

Can be used as a standalone script or imported as a class.
Optionally outputs into versioned MLStorage.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from core.ml_storage import MLStorage


# ======================================================================
# Unified class map for the underwater benchmark datasets
# ======================================================================

UNIFIED_CLASSES: dict[int, str] = {
    0: "crab",
    1: "corals",
    2: "cuttlefish",
    3: "diver",
    4: "echinus",
    5: "fish",
    6: "holothurian",
    7: "jellyfish",
    8: "scallop",
    9: "shrimp",
    10: "small_fish",
    11: "starfish",
    12: "turtle",
    13: "waterweeds",
}

# Maps (original_class_id) → (unified_class_id) per source dataset.
# Built from the data.yaml / ruod.yaml class lists inside each zip.
REMAP_TABLES: dict[str, dict[int, int]] = {
    "Brackish": {0: 0, 1: 5, 2: 7, 3: 9, 4: 10, 5: 11},
    "UOv2":     {0: 4, 1: 6, 2: 8, 3: 11, 4: 13},
    "RUOD":     {0: 6, 1: 4, 2: 8, 3: 11, 4: 5, 5: 1, 6: 3, 7: 2, 8: 12, 9: 7},
}

# Canonical split-name mapping: source folder name → YOLO split name
_SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "valid": "val",
    "val":   "val",
    "test":  "test",
    "testb": "test",
    "testc": "test",
    "testl": "test",
}


# ======================================================================
# YoloIngestor
# ======================================================================


class YoloIngestor:
    """Ingests and merges YOLO-format ZIP datasets with class remapping."""

    def __init__(
        self,
        zip_files: list[str],
        output_dir: str = "dataset",
        storage: MLStorage | None = None,
        version_description: str | None = None,
    ):
        self.zip_paths = [Path(z) for z in zip_files]
        self.output_dir = Path(output_dir)
        self.storage = storage
        self.version_description = version_description

    # ------------------------------------------------------------------
    # Output directory resolution
    # ------------------------------------------------------------------

    def _resolve_output_dir(self) -> Path:
        """Determine the output directory, optionally creating a versioned dataset."""
        if self.storage is not None:
            desc = self.version_description or "yolo_ingest_merged"
            version_dir = self.storage.datasets.create_version(desc)
            return version_dir
        return self.output_dir

    # ------------------------------------------------------------------
    # Locating splits inside an extracted zip
    # ------------------------------------------------------------------

    @staticmethod
    def _find_splits(extracted_root: Path) -> dict[str, list[Path]]:
        """Discover split directories inside an extracted zip.

        Returns a dict mapping YOLO split names (train/val/test) to a list of
        directories that each contain 'images/' and 'labels/' subdirectories.
        Multiple source directories can map to the same split (e.g. RUOD's
        testb, testc, testl all map to 'test').
        Handles both flat and nested (e.g. RUOD/) layouts.
        """
        found: dict[str, list[Path]] = {}

        for alias, yolo_name in _SPLIT_ALIASES.items():
            # Try flat layout first: <root>/<alias>/images/
            flat = extracted_root / alias
            if flat.is_dir() and (flat / "images").is_dir():
                found.setdefault(yolo_name, []).append(flat)
                continue

            # Try one level of nesting: <root>/<subdir>/<alias>/images/
            for subdir in extracted_root.iterdir():
                if not subdir.is_dir() or subdir.name.startswith("."):
                    continue
                nested = subdir / alias
                if nested.is_dir() and (nested / "images").is_dir():
                    found.setdefault(yolo_name, []).append(nested)
                    break

        return found

    # ------------------------------------------------------------------
    # Reading the original data.yaml to auto-detect dataset name
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_dataset_name(extracted_root: Path) -> str | None:
        """Try to identify the dataset from the zip contents."""
        # Check for yaml files that contain the dataset identity
        for pattern in ["data.yaml", "*.yaml"]:
            for yaml_file in extracted_root.rglob(pattern):
                return yaml_file.parent.name if yaml_file.parent != extracted_root else yaml_file.stem
        # Fall back to the first non-hidden subdirectory name
        for subdir in extracted_root.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                return subdir.name
        return None

    @staticmethod
    def _resolve_remap_table(dataset_key: str) -> dict[int, int] | None:
        """Look up a remap table by dataset key (case-insensitive prefix match)."""
        key_lower = dataset_key.lower()
        for name, table in REMAP_TABLES.items():
            if key_lower.startswith(name.lower()) or name.lower().startswith(key_lower):
                return table
        return None

    # ------------------------------------------------------------------
    # Remapping a single label file
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_label_file(
        src_path: Path,
        dst_path: Path,
        remap: dict[int, int],
    ) -> None:
        """Copy a YOLO label file, remapping class IDs."""
        lines = src_path.read_text().strip().splitlines()
        remapped_lines: list[str] = []

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls not in remap:
                # Skip annotations for classes that aren't in the remap table
                continue
            new_cls = remap[old_cls]
            remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        dst_path.write_text("\n".join(remapped_lines) + "\n" if remapped_lines else "")

    # ------------------------------------------------------------------
    # Processing a single zip
    # ------------------------------------------------------------------

    def _process_zip(
        self,
        zip_path: Path,
        images_dir: Path,
        labels_dir: Path,
    ) -> dict[str, int]:
        """Extract one zip, remap labels, and copy into the output dirs.

        Returns a dict of {split_name: image_count}.
        """
        counts: dict[str, int] = {}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            print(f"\nExtracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_path)

            # Detect dataset identity and remap table
            dataset_name = self._detect_dataset_name(tmp_path)
            zip_stem = zip_path.stem  # e.g. "Brackish", "RUOD", "UOv2"

            # Try remap by zip filename first, then by detected name
            remap = self._resolve_remap_table(zip_stem)
            if remap is None and dataset_name:
                remap = self._resolve_remap_table(dataset_name)

            if remap is None:
                print(f"  WARNING: No remap table found for '{zip_stem}' "
                      f"(detected name: {dataset_name}). Skipping this zip.")
                return counts

            print(f"  Dataset: {zip_stem} (detected: {dataset_name})")
            print(f"  Remap table: {remap}")

            splits = self._find_splits(tmp_path)
            if not splits:
                print(f"  WARNING: No valid splits found inside {zip_path.name}. Skipping.")
                return counts

            for yolo_split, split_dirs in splits.items():
                count = 0
                for split_dir in split_dirs:
                    src_images = split_dir / "images"
                    src_labels = split_dir / "labels"

                    if not src_images.exists():
                        continue

                    for img_file in sorted(src_images.iterdir()):
                        if not img_file.is_file() or img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                            continue

                        # Prefix the filename with the dataset name to avoid collisions
                        prefixed_name = f"{zip_stem}_{img_file.name}"
                        dst_img = images_dir / yolo_split / prefixed_name
                        shutil.copy2(img_file, dst_img)

                        # Handle the corresponding label file
                        label_name = img_file.stem + ".txt"
                        src_label = src_labels / label_name
                        dst_label = labels_dir / yolo_split / f"{zip_stem}_{label_name}"

                        if src_label.exists():
                            self._remap_label_file(src_label, dst_label, remap)
                        else:
                            # Create an empty label file (background image)
                            dst_label.write_text("")

                        count += 1

                counts[yolo_split] = count
                print(f"  {yolo_split}: {count} images")

        return counts

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def run(self) -> Path | None:
        """Execute the full ingestion/merge pipeline.

        Returns the path to the output dataset directory, or None on failure.
        """
        # Validate inputs
        for zp in self.zip_paths:
            if not zp.exists():
                print(f"Error: Zip file '{zp}' not found.")
                return None

        effective_output = self._resolve_output_dir()
        images_dir = effective_output / "images"
        labels_dir = effective_output / "labels"

        # Create the YOLO destination structure
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

        # Process each zip
        total_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for zip_path in self.zip_paths:
            per_zip = self._process_zip(zip_path, images_dir, labels_dir)
            for split, count in per_zip.items():
                total_counts[split] = total_counts.get(split, 0) + count

        # Generate dataset.yaml
        yaml_path = effective_output / "dataset.yaml"
        yaml_content = {
            "path": str(effective_output.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": UNIFIED_CLASSES,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"\n{'='*60}")
        print(f"Merged dataset created at: {effective_output}")
        print(f"  dataset.yaml: {yaml_path}")
        print(f"  Classes: {len(UNIFIED_CLASSES)}")
        for split, count in total_counts.items():
            print(f"  {split}: {count} images")
        print(f"{'='*60}")

        return effective_output


# ======================================================================
# CLI entry point
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest and merge YOLO-format zip datasets with class remapping."
    )
    parser.add_argument(
        "--zip-files", nargs="+", required=True,
        help="Paths to the .zip files containing YOLO datasets"
    )
    parser.add_argument(
        "--output-dir", default="dataset", type=str,
        help="Output directory for the merged YOLO dataset (default: dataset)"
    )
    # ML Storage options
    parser.add_argument(
        "--use-storage", action="store_true",
        help="Enable versioned dataset storage via MLStorage"
    )
    parser.add_argument(
        "--storage-root", type=str, default="ml_storage",
        help="Root directory for MLStorage (default: ml_storage)"
    )
    parser.add_argument(
        "--version-desc", type=str, default=None,
        help="Description for this dataset version (e.g. 'merged underwater benchmark')"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    storage = None
    if args.use_storage:
        storage = MLStorage(args.storage_root)

    ingestor = YoloIngestor(
        zip_files=args.zip_files,
        output_dir=args.output_dir,
        storage=storage,
        version_description=args.version_desc,
    )
    ingestor.run()


if __name__ == "__main__":
    main()

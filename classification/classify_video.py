#!/usr/bin/env python3
"""
Classify the position of every tracked fish in every frame of a crops/<video>/ folder
and emit one Excel file with per-fish percentages plus an averaged row.

Temporal smoothing: majority vote in a sliding window (default 5 frames) over each
fish's per-frame predictions. Cheap, no extra model needed.

Weights can be supplied directly (--weights path) or resolved from the MLStorage
classifier registry (--use-storage, with optional --run to pick a specific run).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from collections import Counter

import pandas as pd
from ultralytics import YOLO
from core.ml_storage import MLStorage

DEFAULT_MODEL_NAME = "fish_position_classifier"

def smooth(labels: list[str], window: int) -> list[str]:
    if window <= 1 or len(labels) <= 1:
        return labels
    half = window // 2
    out = []
    for i in range(len(labels)):
        chunk = labels[max(0, i - half): i + half + 1]
        out.append(Counter(chunk).most_common(1)[0][0])
    return out


def frame_index(path: Path) -> int:
    # filenames look like "frame_123.jpg"
    return int(path.stem.split("_")[-1])


def classify_fish_folder(model: YOLO, fish_dir: Path, device: str) -> list[str]:
    crops = sorted(fish_dir.glob("frame_*.jpg"), key=frame_index)
    if not crops:
        return []
    results = model.predict(source=[str(c) for c in crops], device=device, verbose=False)
    names = model.names
    return [names[int(r.probs.top1)] for r in results]


def _resolve_weights(args) -> Path:
    """Return a validated weights path from --weights or the classifier registry."""
    if args.use_storage:
        storage = MLStorage(args.storage_root)
        weights_path = storage.classifier_models.get_weights(
            model_name=args.model_name,
            run=args.run,
        )
        label = f"run {args.run}" if args.run else "latest run"
        print(f"Resolved weights from registry ({label}): {weights_path}")
        return weights_path

    if args.weights is None:
        raise FileNotFoundError(
            "No weights specified. Provide --weights or use --use-storage."
        )
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file '{weights_path}' not found. "
            "Provide a valid --weights path or use --use-storage."
        )
    return weights_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", required=True,
                        help="crops/<video_name>/ produced by extract_crops.py")
    parser.add_argument("--weights", default=None,
                        help="Trained classifier weights (best.pt)")
    parser.add_argument("--output", default=None,
                        help="Output xlsx path (default: <crops-dir>.xlsx)")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--reject-class", default="unclear",
                        help="Class name to exclude from reported percentages")
    parser.add_argument("--device", default="mps")
    # ML Storage options
    parser.add_argument("--use-storage", action="store_true",
                        help="Fetch weights from the MLStorage classifier registry")
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model name in the registry (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--run", type=int, default=None,
                        help="1-indexed run number from the registry (default: latest)")
    args = parser.parse_args()

    try:
        weights_path = _resolve_weights(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    crops_dir = Path(args.crops_dir)
    fish_dirs = sorted([p for p in crops_dir.iterdir() if p.is_dir() and p.name.startswith("id_")])
    if not fish_dirs:
        raise SystemExit(f"No id_* subfolders in {crops_dir}")

    model = YOLO(weights_path)
    class_names = list(model.names.values())
    reported_classes = [c for c in class_names if c != args.reject_class]

    rows = []
    for fish_dir in fish_dirs:
        raw = classify_fish_folder(model, fish_dir, args.device)
        if not raw:
            continue
        smoothed = smooth(raw, args.smooth_window)
        counts = Counter(smoothed)
        scored = sum(counts.get(c, 0) for c in reported_classes)
        if scored == 0:
            continue
        row = {
            "fish_id": fish_dir.name,
            "n_frames": len(smoothed),
            "n_scored": scored,
            "n_rejected": counts.get(args.reject_class, 0),
        }
        for cls in reported_classes:
            row[cls] = 100.0 * counts.get(cls, 0) / scored
        rows.append(row)

    df = pd.DataFrame(rows)

    # Averaged row: mean of per-fish percentages across all fish in the video.
    avg = {
        "fish_id": "AVERAGE",
        "n_frames": df["n_frames"].sum(),
        "n_scored": df["n_scored"].sum(),
        "n_rejected": df["n_rejected"].sum(),
    }
    for cls in reported_classes:
        avg[cls] = df[cls].mean()
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    out_path = Path(args.output) if args.output else crops_dir.with_suffix(".xlsx")
    df.to_excel(out_path, index=False)
    print(f"Wrote {out_path} ({len(rows)} fish, {len(class_names)} classes)")


if __name__ == "__main__":
    main()

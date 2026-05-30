#!/usr/bin/env python3
"""
Classify the position of every tracked fish in every frame of a crops/<video>/ folder
and emit one Excel file with per-fish percentages plus an averaged row.

Temporal smoothing: majority vote in a sliding window (default 5 frames) over each
fish's per-frame predictions. Cheap, no extra model needed.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", required=True,
                        help="crops/<video_name>/ produced by extract_crops.py")
    parser.add_argument("--weights", required=True,
                        help="Trained classifier weights (best.pt from train_classifier.py)")
    parser.add_argument("--output", default=None,
                        help="Output xlsx path (default: <crops-dir>.xlsx)")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--reject-class", default="unclear",
                        help="Class name to exclude from reported percentages")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    fish_dirs = sorted([p for p in crops_dir.iterdir() if p.is_dir() and p.name.startswith("id_")])
    if not fish_dirs:
        raise SystemExit(f"No id_* subfolders in {crops_dir}")

    model = YOLO(args.weights)
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

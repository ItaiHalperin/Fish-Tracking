"""
Run a trained classifier over a video's crops folder and emit an Excel
report: one row per tracked fish, plus an averaged row.

Usage:
    python -m classifier.infer --run results/<name> --crops-dir crops/<video>
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import load_config
from .models import build_model


def smooth(labels: list[str], window: int) -> list[str]:
    if window <= 1 or len(labels) <= 1:
        return labels
    half = window // 2
    out = []
    for i in range(len(labels)):
        chunk = labels[max(0, i - half): i + half + 1]
        out.append(Counter(chunk).most_common(1)[0][0])
    return out


def frame_index(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def classify_fish(model, fish_dir: Path) -> list[str]:
    crops = sorted(fish_dir.glob("frame_*.jpg"), key=frame_index)
    if not crops:
        return []
    return [p[0] for p in model.predict(crops)]


def run(run_dir: Path, crops_dir: Path, smooth_window: int = 5,
        reject_class: str = "unclear", output: Path | None = None) -> Path:
    cfg = load_config(run_dir / "config.yaml")
    weights = run_dir / "weights" / "best.pt"

    model = build_model(cfg.model_type)
    model.load(weights)

    reported = [c for c in model.class_names if c != reject_class]

    fish_dirs = sorted(p for p in crops_dir.iterdir()
                       if p.is_dir() and p.name.startswith("id_"))
    if not fish_dirs:
        raise SystemExit(f"No id_* folders in {crops_dir}")

    rows = []
    for fish_dir in fish_dirs:
        raw = classify_fish(model, fish_dir)
        if not raw:
            continue
        labels = smooth(raw, smooth_window)
        counts = Counter(labels)
        scored = sum(counts.get(c, 0) for c in reported)
        if scored == 0:
            continue
        row = {
            "fish_id": fish_dir.name,
            "n_frames": len(labels),
            "n_scored": scored,
            "n_rejected": counts.get(reject_class, 0),
        }
        for cls in reported:
            row[cls] = 100.0 * counts.get(cls, 0) / scored
        rows.append(row)

    df = pd.DataFrame(rows)
    avg = {
        "fish_id": "AVERAGE",
        "n_frames": df["n_frames"].sum(),
        "n_scored": df["n_scored"].sum(),
        "n_rejected": df["n_rejected"].sum(),
    }
    for cls in reported:
        avg[cls] = df[cls].mean()
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    out_path = output or crops_dir.with_suffix(".xlsx")
    df.to_excel(out_path, index=False)
    print(f"Wrote {out_path} ({len(rows)} fish, {len(reported)} classes)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to results/<name>")
    parser.add_argument("--crops-dir", required=True, help="crops/<video_name>/")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--reject-class", default="unclear")
    args = parser.parse_args()

    run(Path(args.run), Path(args.crops_dir),
        smooth_window=args.smooth_window,
        reject_class=args.reject_class,
        output=Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One-time seeding: for every class that's currently under-represented in
labels_raw/, find the unlabeled neighbors (same video + same fish ID) of
each already-labeled crop in that class, write them to a file.

Run once, then launch the webapp with `--seed-queue <output>` so those
seed crops surface before random sampling kicks in.

    python webapp/seed_rare_neighbors.py \\
        --raw labels_raw \\
        --source crops/all_flat \\
        --threshold 30 \\
        --neighbors-per 10 \\
        --output seed_queue.txt
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

FLAT_PATTERN = re.compile(r"^(?P<video>.+)__id_(?P<fid>\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$")


def parse(name: str) -> dict | None:
    m = FLAT_PATTERN.match(name)
    return m.groupdict() if m else None


def neighbors(source: Path, video: str, fid: str, frame: int, n: int, taken: set[str]) -> list[str]:
    pattern = f"{video}__id_{fid}_frame_*.jpg"
    hits = []
    for p in source.glob(pattern):
        if p.name in taken:
            continue
        m = parse(p.name)
        if not m:
            continue
        hits.append((abs(int(m["frame"]) - frame), p.name))
    hits.sort()
    return [name for _, name in hits[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="labels_raw")
    parser.add_argument("--source", default="crops/all_flat")
    parser.add_argument("--threshold", type=int, default=30,
                        help="Classes with fewer than this many labels get seeded")
    parser.add_argument("--neighbors-per", type=int, default=10,
                        help="How many same-fish neighbors per labeled crop")
    parser.add_argument("--skip-class", action="append", default=["unclear"],
                        help="Classes to ignore (default: unclear)")
    parser.add_argument("--output", default="seed_queue.txt")
    args = parser.parse_args()

    raw = Path(args.raw)
    source = Path(args.source)
    if not raw.is_dir() or not source.is_dir():
        raise SystemExit("Missing --raw or --source dir")

    # Identify rare classes
    rare_classes = []
    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        if cls_dir.name in args.skip_class:
            continue
        count = sum(1 for _ in cls_dir.iterdir())
        if count < args.threshold and count > 0:
            rare_classes.append((cls_dir, count))

    if not rare_classes:
        print(f"No classes below {args.threshold} samples (excluding {args.skip_class}). Nothing to seed.")
        return

    print(f"Seeding for {len(rare_classes)} rare classes:")
    for cls_dir, count in rare_classes:
        print(f"  {count:3d}  {cls_dir.name}")

    seed: list[str] = []
    taken: set[str] = set()  # avoid dupes across rare classes
    per_class_added: Counter = Counter()
    skipped_unparseable = 0

    for cls_dir, _ in rare_classes:
        for f in sorted(cls_dir.iterdir()):
            m = parse(f.name)
            if not m:
                skipped_unparseable += 1
                continue
            ns = neighbors(source, m["video"], m["fid"], int(m["frame"]),
                           args.neighbors_per, taken)
            for n in ns:
                if n in taken:
                    continue
                taken.add(n)
                seed.append(n)
                per_class_added[cls_dir.name] += 1

    Path(args.output).write_text("\n".join(seed) + "\n")
    print(f"\nWrote {len(seed)} seed filenames to {args.output}")
    print(f"By rare class (unlabeled neighbors found):")
    for cls, n in per_class_added.most_common():
        print(f"  {n:4d}  {cls}")
    if skipped_unparseable:
        print(f"\n  (skipped {skipped_unparseable} labeled filenames that didn't match the flat naming pattern)")


if __name__ == "__main__":
    main()

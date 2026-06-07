#!/usr/bin/env python3
"""
Flatten crops/<video>/id_*/frame_*.jpg into one folder for easy hand-labeling.
Filenames keep the id and frame number so we can trace any crop back.
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", required=True,
                        help="crops/<video_name>/ produced by extract_crops.py")
    parser.add_argument("--output-dir", default=None,
                        help="Destination folder (default: <crops-dir>_flat)")
    parser.add_argument("--move", action="store_true",
                        help="Move instead of copy (saves disk, destroys originals)")
    args = parser.parse_args()

    src = Path(args.crops_dir)
    dst = Path(args.output_dir) if args.output_dir else src.with_name(src.name + "_flat")
    dst.mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2
    count = 0
    for fish_dir in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("id_")):
        for crop in sorted(fish_dir.glob("frame_*.jpg")):
            out_name = f"{fish_dir.name}_{crop.name}"
            op(str(crop), str(dst / out_name))
            count += 1

    print(f"{'Moved' if args.move else 'Copied'} {count} crops into {dst}")


if __name__ == "__main__":
    main()

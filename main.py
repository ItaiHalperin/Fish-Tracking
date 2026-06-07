#!/usr/bin/env python3
"""
Video Frame Sampler.

This script scans a directory of videos, randomly samples a user-specified
number of frames from each video, and saves them to a single 'raw_frames' folder
so you can easily upload them to an annotation tool.
"""

import argparse
import random
from pathlib import Path
from video_utils import find_video_files, gather_frame_metadata, extract_and_save_frames

def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample frames from videos and save them to a single folder."
    )
    parser.add_argument("--videos-dir", type=str, default="videos", help="Path to the directory containing video files (default: videos).")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Path to the output directory (default: dataset).")
    parser.add_argument("--num-frames", type=int, default=50, help="Number of frames to randomly select from each video (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    if not videos_dir.exists():
        print(f"Error: Videos directory '{videos_dir}' does not exist.")
        return

    # 1. Discover videos
    video_files = find_video_files(videos_dir)
    if not video_files:
        print(f"No video files found in '{videos_dir}'.")
        return

    print(f"Found {len(video_files)} video files in '{videos_dir}'.")

    # 2. Gather metadata
    all_frames_metadata = gather_frame_metadata(video_files, args.num_frames)
    if not all_frames_metadata:
        print("No frames were successfully gathered. Exiting.")
        return

    print(f"\nTotal sampled frames: {len(all_frames_metadata)}")

    # 3. Build extraction plan
    # extraction_plan is a dict mapping video_path to list of (frame_idx, timestamp)
    extraction_plan = {}
    for item in all_frames_metadata:
        extraction_plan.setdefault(item["video_path"], []).append(
            (item["frame_idx"], item["timestamp"])
        )

    # 4. Create destination directory
    raw_frames_dir = output_dir / "raw_frames"
    raw_frames_dir.mkdir(parents=True, exist_ok=True)

    # 5. Perform extraction
    print("\nStarting frame extraction...")
    extracted_count = extract_and_save_frames(extraction_plan, raw_frames_dir)

    print(f"\nSuccess! Extracted and saved {extracted_count} frames to '{raw_frames_dir}'.")

if __name__ == "__main__":
    main()

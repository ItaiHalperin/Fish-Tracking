#!/usr/bin/env python3
"""
Cloud Video Frame Sampler via rclone.

This script lists videos from a remote SharePoint/OneDrive directory, randomly
selects a few, downloads them temporarily, extracts random frames, and then
cleans up the downloaded videos to save hard drive space.
"""

import argparse
import random
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from video_utils import gather_frame_metadata, extract_and_save_frames

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample frames from cloud videos via rclone.")
    parser.add_argument("--remote-dir", required=True, type=str, help="Rclone remote directory (e.g., sharepoint:Videos)")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Path to output directory (default: dataset)")
    parser.add_argument("--num-videos", type=int, default=3, help="Number of random videos to download and process (default: 3)")
    parser.add_argument("--num-frames", type=int, default=50, help="Number of frames to extract per video (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()

def list_remote_videos(remote_dir: str):
    """Lists video files inside an rclone remote directory."""
    print(f"Querying remote directory (including subfolders): {remote_dir} ...")
    # Adding -R to recursively search all subdirectories inside the main folder
    cmd = ["rclone", "lsjson", "-R", remote_dir]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running rclone: {e.stderr}")
        return []
        
    try:
        files = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Error parsing JSON output from rclone.")
        return []
        
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm", ".mts"}
    videos = []
    for f in files:
        if f.get("IsDir", False):
            continue
        path = f.get("Path", "")
        if any(path.lower().endswith(ext) for ext in video_extensions):
            videos.append(path)
            
    return videos

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # 1. Get list of remote videos
    remote_videos = list_remote_videos(args.remote_dir)
    if not remote_videos:
        print(f"No videos found in remote directory '{args.remote_dir}'. Check your path.")
        return
        
    print(f"Found {len(remote_videos)} video(s) in the cloud.")
    
    # 2. Select random videos
    num_to_select = min(args.num_videos, len(remote_videos))
    selected_videos = random.sample(remote_videos, num_to_select)
    print(f"\nRandomly selected {num_to_select} video(s) for extraction:")
    for v in selected_videos:
        print(f" - {v}")
        
    # 3. Download the videos temporarily
    temp_dir = Path(tempfile.mkdtemp(prefix="rclone_videos_"))
    print(f"\nDownloading videos to temporary directory: {temp_dir} ...")
    
    for v in selected_videos:
        remote_path = f"{args.remote_dir}/{v}"
        cmd = ["rclone", "copy", remote_path, str(temp_dir), "--progress"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to download {v}. Skipping.")
            
    # 4. Extract frames using our existing pipeline
    local_videos = list(temp_dir.glob("*"))
    if not local_videos:
        print("No videos were successfully downloaded.")
        shutil.rmtree(temp_dir)
        return
        
    print("\nGathering frame metadata...")
    all_frames_metadata = gather_frame_metadata(local_videos, args.num_frames)
    
    if not all_frames_metadata:
        print("No frames could be read. Cleaning up.")
        shutil.rmtree(temp_dir)
        return
        
    extraction_plan = {}
    for item in all_frames_metadata:
        extraction_plan.setdefault(item["video_path"], []).append(
            (item["frame_idx"], item["timestamp"])
        )
        
    output_dir = Path(args.output_dir)
    raw_frames_dir = output_dir / "raw_frames"
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting {len(all_frames_metadata)} frames...")
    extracted_count = extract_and_save_frames(extraction_plan, raw_frames_dir)
    
    # 5. Cleanup
    print("\nCleaning up downloaded videos to save space...")
    shutil.rmtree(temp_dir)
    
    print(f"\nSuccess! Downloaded {len(local_videos)} videos, extracted {extracted_count} frames to '{raw_frames_dir}', and deleted the videos.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cloud Video Frame Sampler via rclone.

This script lists videos from a remote directory, randomly selects some,
downloads and processes them ONE AT A TIME, and only assigns train/val/test
splits AFTER all videos are processed — so corrupted videos never leave
a split with missing data.
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
    parser.add_argument("--remote-dir", required=True, type=str, help="Rclone remote directory (e.g., gdrive:GOLDIES)")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Path to output directory (default: dataset)")
    parser.add_argument("--num-videos", type=int, default=27, help="Number of random videos to process (default: 27)")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract per video (default: 10)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of videos for validation (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of videos for testing (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()

def list_remote_videos(remote_dir: str):
    """Lists video files inside an rclone remote directory (recursively)."""
    print(f"Querying remote directory (including subfolders): {remote_dir} ...")
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

def convert_mts_if_needed(video_path: Path) -> Path:
    """Converts .MTS files to .mp4 using FFmpeg. Returns the path to the usable file."""
    if video_path.suffix.lower() != ".mts":
        return video_path
        
    mp4_path = video_path.with_suffix(".mp4")
    print(f"  Remuxing {video_path.name} → {mp4_path.name} ...")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", "-an", str(mp4_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        video_path.unlink()
        return mp4_path
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Failed to convert {video_path.name}.")
        print(f"  FFmpeg error: {e.stderr[-500:] if e.stderr else 'unknown'}")
        return None

def process_single_video(remote_dir: str, remote_path: str, staging_dir: Path, num_frames: int) -> str | None:
    """Downloads a single video, extracts frames to staging, and cleans up.
    Returns the video name if successful, None if failed."""
    temp_dir = Path(tempfile.mkdtemp(prefix="rclone_single_"))
    
    try:
        # Download
        filename = Path(remote_path).name
        remote_parent = f"{remote_dir}/{str(Path(remote_path).parent)}" if "/" in remote_path else remote_dir
        cmd = ["rclone", "copy", remote_parent, str(temp_dir), "--include", filename, "--progress"]
        subprocess.run(cmd, check=True)
        
        # Find the downloaded file
        local_files = [f for f in temp_dir.iterdir() if f.is_file()]
        if not local_files:
            print("  Download failed — no file found.")
            return None
            
        local_video = local_files[0]
        
        # Convert MTS if needed
        local_video = convert_mts_if_needed(local_video)
        if local_video is None:
            return None
        
        # Generate a unique stem based on its remote path to prevent naming collisions
        # e.g., "RGoldies 10_6_25/00057.MTS" -> "RGoldies_10_6_25_00057"
        unique_stem = Path(remote_path).with_suffix("").as_posix().replace("/", "_").replace(" ", "_")
        unique_video_path = local_video.with_name(unique_stem + local_video.suffix)
        local_video.rename(unique_video_path)
        local_video = unique_video_path
        
        # Extract frames into staging_dir/{video_name}/
        metadata = gather_frame_metadata([local_video], num_frames)
        if not metadata:
            print("  Could not read any frames from this video.")
            return None
            
        extraction_plan = {}
        for item in metadata:
            extraction_plan.setdefault(item["video_path"], []).append(
                (item["frame_idx"], item["timestamp"])
            )
            
        extracted = extract_and_save_frames(extraction_plan, staging_dir)
        if extracted == 0:
            return None
            
        return local_video.stem
        
    except subprocess.CalledProcessError:
        print("  Download failed.")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # 1. List all remote videos
    remote_videos = list_remote_videos(args.remote_dir)
    if not remote_videos:
        print(f"No videos found in '{args.remote_dir}'. Check your path.")
        return
        
    print(f"Found {len(remote_videos)} video(s) in the cloud.\n")
    
    # 2. Select random videos
    num_to_select = min(args.num_videos, len(remote_videos))
    selected = random.sample(remote_videos, num_to_select)
    
    # 3. Create a staging directory for all frames (before splitting)
    output_dir = Path(args.output_dir)
    staging_dir = output_dir / "raw_frames" / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Download and process videos one at a time into staging
    total = len(selected)
    successful_videos = []
    
    for i, video_path in enumerate(selected, 1):
        unique_name = Path(video_path).with_suffix("").as_posix().replace("/", "_").replace(" ", "_")
        print(f"\n[{i}/{total}] Downloading '{video_path}' (as '{unique_name}') ...")
        
        result = process_single_video(args.remote_dir, video_path, staging_dir, args.num_frames)
        
        if result:
            successful_videos.append(result)
            frame_count = len(list((staging_dir / result).glob("*.jpg")))
            print(f"[{i}/{total}] ✓ Success — {frame_count} frames (total videos OK: {len(successful_videos)})")
        else:
            print(f"[{i}/{total}] ✗ Skipped (corrupted or unreadable)")
    
    if len(successful_videos) < 3:
        print(f"\nError: Only {len(successful_videos)} video(s) succeeded. Need at least 3 for train/val/test split.")
        shutil.rmtree(staging_dir)
        return
    
    # 5. NOW do the video-level split — only on videos that actually worked
    random.shuffle(successful_videos)
    num_val = max(1, int(round(len(successful_videos) * args.val_ratio)))
    num_test = max(1, int(round(len(successful_videos) * args.test_ratio)))
    
    if num_val + num_test >= len(successful_videos):
        num_val = 1
        num_test = 1
    
    val_videos = successful_videos[:num_val]
    test_videos = successful_videos[num_val:num_val + num_test]
    train_videos = successful_videos[num_val + num_test:]
    
    print(f"\n{'='*60}")
    print(f"Video-level split (on {len(successful_videos)} successful videos):")
    print(f"  Train: {len(train_videos)} videos — {train_videos}")
    print(f"  Val:   {len(val_videos)} videos — {val_videos}")
    print(f"  Test:  {len(test_videos)} videos — {test_videos}")
    
    # 6. Move frame folders from staging into the correct split directories
    raw_frames_dir = output_dir / "raw_frames"
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    for split_name, video_list in [("train", train_videos), ("val", val_videos), ("test", test_videos)]:
        split_dir = raw_frames_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for video_name in video_list:
            src = staging_dir / video_name
            dst = split_dir / video_name
            if src.exists():
                shutil.move(str(src), str(dst))
                frame_count = len(list(dst.glob("*.jpg")))
                split_counts[split_name] += frame_count
    
    # 7. Cleanup staging
    shutil.rmtree(staging_dir, ignore_errors=True)
    
    print(f"\nFinal frame counts:")
    print(f"  Train: {split_counts['train']} frames")
    print(f"  Val:   {split_counts['val']} frames")
    print(f"  Test:  {split_counts['test']} frames")
    total_frames = sum(split_counts.values())
    print(f"  Total: {total_frames} frames")
    print(f"\nOutput: {raw_frames_dir}")

if __name__ == "__main__":
    main()

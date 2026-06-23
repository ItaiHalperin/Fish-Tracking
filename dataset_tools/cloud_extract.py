#!/usr/bin/env python3
"""
Cloud Video Frame Sampler via rclone.

This module provides a CloudExtractor class that lists videos from a remote
directory, randomly selects some, downloads and processes them ONE AT A TIME,
and only assigns train/val/test splits AFTER all videos are processed — so
corrupted videos never leave a split with missing data.

Can be used as a standalone script (python cloud_extract.py ...) or imported
as a class in other modules.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime
import random
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from core.video_utils import gather_frame_metadata, extract_and_save_frames
from core.ml_storage import MLStorage

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm", ".mts"}


class CloudExtractor:
    """Extracts random frames from videos stored on an rclone remote."""

    def __init__(
        self,
        remote_dir: str,
        output_dir: str = "dataset",
        num_videos: int = 27,
        num_frames: int = 10,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        storage: MLStorage | None = None,
        version_description: str | None = None,
    ):
        self.remote_dir = remote_dir
        self.output_dir = Path(output_dir)
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.storage = storage
        self.version_description = version_description

    # ------------------------------------------------------------------
    # Remote listing
    # ------------------------------------------------------------------

    def list_remote_videos(self) -> list[str]:
        """Lists video files inside the rclone remote directory (recursively)."""
        print(f"Querying remote directory (including subfolders): {self.remote_dir} ...")
        cmd = ["rclone", "lsjson", "-R", self.remote_dir]
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

        videos = []
        for f in files:
            if f.get("IsDir", False):
                continue
            path = f.get("Path", "")
            if any(path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                videos.append(path)

        return videos

    # ------------------------------------------------------------------
    # MTS conversion
    # ------------------------------------------------------------------

    @staticmethod
    def convert_mts_if_needed(video_path: Path) -> Path | None:
        """Converts .MTS files to .mp4 using FFmpeg. Returns the path to the usable file."""
        if video_path.suffix.lower() != ".mts":
            return video_path

        mp4_path = video_path.with_suffix(".mp4")
        print(f"  Remuxing {video_path.name} → {mp4_path.name} ...")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", "-an", str(mp4_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            video_path.unlink()
            return mp4_path
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to convert {video_path.name}.")
            print(f"  FFmpeg error: {e.stderr[-500:] if e.stderr else 'unknown'}")
            return None

    # ------------------------------------------------------------------
    # Single-video processing
    # ------------------------------------------------------------------

    def process_single_video(self, remote_path: str, staging_dir: Path) -> str | None:
        """Downloads a single video, extracts frames to staging, and cleans up.
        Returns the video name if successful, None if failed."""
        temp_dir = Path(tempfile.mkdtemp(prefix="rclone_single_"))

        try:
            # Download
            filename = Path(remote_path).name
            remote_parent = (
                f"{self.remote_dir}/{str(Path(remote_path).parent)}"
                if "/" in remote_path
                else self.remote_dir
            )
            cmd = ["rclone", "copy", remote_parent, str(temp_dir), "--include", filename, "--progress"]
            subprocess.run(cmd, check=True)

            # Find the downloaded file
            local_files = [f for f in temp_dir.iterdir() if f.is_file()]
            if not local_files:
                print("  Download failed — no file found.")
                return None

            local_video = local_files[0]

            # Convert MTS if needed
            local_video = self.convert_mts_if_needed(local_video)
            if local_video is None:
                return None

            # Generate a unique stem based on its remote path to prevent naming collisions
            # e.g., "RGoldies 10_6_25/00057.MTS" -> "RGoldies_10_6_25_00057"
            unique_stem = Path(remote_path).with_suffix("").as_posix().replace("/", "_").replace(" ", "_")
            unique_video_path = local_video.with_name(unique_stem + local_video.suffix)
            local_video.rename(unique_video_path)
            local_video = unique_video_path

            # Extract frames into staging_dir/{video_name}/
            metadata = gather_frame_metadata([local_video], self.num_frames)
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

    # ------------------------------------------------------------------
    # Split & move
    # ------------------------------------------------------------------

    def _split_and_move(self, successful_videos: list[str], staging_dir: Path) -> None:
        """Assigns train/val/test splits and moves frame folders out of staging."""
        random.shuffle(successful_videos)
        num_val = max(1, int(round(len(successful_videos) * self.val_ratio)))
        num_test = max(1, int(round(len(successful_videos) * self.test_ratio)))

        if num_val + num_test >= len(successful_videos):
            num_val = 1
            num_test = 1

        val_videos = successful_videos[:num_val]
        test_videos = successful_videos[num_val : num_val + num_test]
        train_videos = successful_videos[num_val + num_test :]

        print(f"\n{'='*60}")
        print(f"Video-level split (on {len(successful_videos)} successful videos):")
        print(f"  Train: {len(train_videos)} videos — {train_videos}")
        print(f"  Val:   {len(val_videos)} videos — {val_videos}")
        print(f"  Test:  {len(test_videos)} videos — {test_videos}")

        raw_frames_dir = self.output_dir / "raw_frames"
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

        # Cleanup staging
        shutil.rmtree(staging_dir, ignore_errors=True)

        print(f"\nFinal frame counts:")
        print(f"  Train: {split_counts['train']} frames")
        print(f"  Val:   {split_counts['val']} frames")
        print(f"  Test:  {split_counts['test']} frames")
        total_frames = sum(split_counts.values())
        print(f"  Total: {total_frames} frames")
        print(f"\nOutput: {raw_frames_dir}")

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def _resolve_output_dir(self) -> Path:
        """Determine the output directory, optionally creating a versioned dataset."""
        if self.storage is not None:
            desc = self.version_description or f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version_dir = self.storage.datasets.create_version(desc)
            return version_dir
        return self.output_dir

    def run(self) -> None:
        """Execute the full extraction pipeline."""
        random.seed(self.seed)

        # 1. List all remote videos
        remote_videos = self.list_remote_videos()
        if not remote_videos:
            print(f"No videos found in '{self.remote_dir}'. Check your path.")
            return

        print(f"Found {len(remote_videos)} video(s) in the cloud.\n")

        # 2. Select random videos
        num_to_select = min(self.num_videos, len(remote_videos))
        selected = random.sample(remote_videos, num_to_select)

        # 3. Resolve output directory (plain or versioned)
        effective_output = self._resolve_output_dir()

        # 4. Create a staging directory for all frames (before splitting)
        staging_dir = effective_output / "raw_frames" / "_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # 5. Download and process videos one at a time into staging
        total = len(selected)
        successful_videos = []

        for i, video_path in enumerate(selected, 1):
            unique_name = Path(video_path).with_suffix("").as_posix().replace("/", "_").replace(" ", "_")
            print(f"\n[{i}/{total}] Downloading '{video_path}' (as '{unique_name}') ...")

            result = self.process_single_video(video_path, staging_dir)

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

        # 6. Split and move frames
        # Temporarily point output_dir to effective_output for _split_and_move
        original_output = self.output_dir
        self.output_dir = effective_output
        self._split_and_move(successful_videos, staging_dir)
        self.output_dir = original_output


# ======================================================================
# CLI entry point
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample frames from cloud videos via rclone.")
    parser.add_argument("--remote-dir", required=True, type=str, help="Rclone remote directory (e.g., gdrive:GOLDIES)")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Path to output directory (default: dataset)")
    parser.add_argument("--num-videos", type=int, default=27, help="Number of random videos to process (default: 27)")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract per video (default: 10)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of videos for validation (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of videos for testing (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    # ML Storage options
    parser.add_argument("--use-storage", action="store_true",
                        help="Enable versioned dataset storage via MLStorage")
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--version-desc", type=str, default=None,
                        help="Description for this dataset version (e.g. 'added night footage')")
    return parser.parse_args()


def main():
    args = parse_args()

    storage = None
    if args.use_storage:
        storage = MLStorage(args.storage_root)

    extractor = CloudExtractor(
        remote_dir=args.remote_dir,
        output_dir=args.output_dir,
        num_videos=args.num_videos,
        num_frames=args.num_frames,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        storage=storage,
        version_description=args.version_desc,
    )
    extractor.run()


if __name__ == "__main__":
    main()

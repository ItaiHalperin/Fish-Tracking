#!/usr/bin/env python3
"""
Goldfish Tracking Script
Runs the trained YOLOv11 model on a video to detect and track goldfish across frames.

Weights can be specified directly (--weights path) or fetched from the
MLStorage model registry (--use-storage, with optional --run to pick a
specific run). Track outputs are NOT uploaded to ml_storage.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
import tempfile
from pathlib import Path
from ultralytics import YOLO
from core.ml_storage import MLStorage

DEFAULT_MODEL_NAME = "goldfish_yolo"


def _resolve_weights(args) -> Path:
    """Return a validated weights path from either --weights or the model registry."""
    if args.use_storage:
        storage = MLStorage(args.storage_root)
        run_num = args.run  # None means latest
        weights_path = storage.detection_models.get_weights(
            model_name=args.model_name,
            run=run_num,
        )
        label = f"run {run_num}" if run_num else "latest run"
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
    parser = argparse.ArgumentParser(description="Track goldfish in a video using the trained YOLOv11 model.")
    parser.add_argument("--video", required=True, type=str, help="Path to the input video (.mp4, .MTS, etc.)")
    parser.add_argument("--output", required=True, type=str, help="Output directory for tracking results")
    parser.add_argument("--weights", default=None,
                        type=str, help="Path to the trained best.pt weights")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"], 
                        help="Tracking algorithm to use")
    parser.add_argument("--start", type=int, default=0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=int, default=None, 
                        help="Only process N seconds of the video (from --start)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    # ML Storage options
    parser.add_argument("--use-storage", action="store_true",
                        help="Fetch weights from the MLStorage model registry instead of --weights")
    parser.add_argument("--storage-root", type=str, default="ml_storage",
                        help="Root directory for MLStorage (default: ml_storage)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Model name in the registry (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--run", type=int, default=None,
                        help="1-indexed run number from the registry (default: latest)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found.")
        return

    try:
        weights_path = _resolve_weights(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Handle duration cropping
    temp_file = None
    target_video = video_path
    
    if args.duration is not None or args.start > 0:
        start_str = str(args.start)
        desc = f"from {args.start}s"
        if args.duration:
            desc += f" for {args.duration}s"
        print(f"Extracting clip {desc} using FFmpeg...")
        temp_dir = tempfile.gettempdir()
        temp_file = Path(temp_dir) / f"trimmed_{video_path.stem}.mp4"
        
        # We use -c copy to instantly copy the streams without re-encoding
        cmd = [
            "ffmpeg", "-y", "-ss", start_str, "-i", str(video_path),
        ]
        if args.duration:
            cmd += ["-t", str(args.duration)]
        cmd += ["-c", "copy", str(temp_file)]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            target_video = temp_file
            print("Cropping successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error cropping video with FFmpeg: {e}")
            return

    print(f"Loading YOLOv11 model from {weights_path}...")
    model = YOLO(weights_path)
    
    print(f"Starting tracking on {target_video.name} using {args.tracker} (conf={args.conf})...")

    output_path = Path(args.output)
    results = model.track(
        source=str(target_video),
        tracker=args.tracker,
        save=True,       
        conf=args.conf,  
        device="mps",
        project=str(output_path.parent),
        name=output_path.name,
    )
    
    # Cleanup temp file if we created one
    if temp_file and temp_file.exists():
        temp_file.unlink()
        
    print(f"\nTracking complete! Results saved to '{output_path}'.")

if __name__ == "__main__":
    main()

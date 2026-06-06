#!/usr/bin/env python3
"""
Goldfish Tracking Script
Runs the trained YOLOv11 model on a video to detect and track goldfish across frames.
"""

import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Track goldfish in a video using the trained YOLOv11 model.")
    parser.add_argument("--video", required=True, type=str, help="Path to the input video (.mp4, .MTS, etc.)")
    parser.add_argument("--weights", default="runs/detect/runs/detect/fish_tracking_model-3/weights/best.pt",
                        type=str, help="Path to the trained best.pt weights")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"], 
                        help="Tracking algorithm to use")
    parser.add_argument("--start", type=int, default=0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=int, default=None, 
                        help="Only process N seconds of the video (from --start)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found.")
        return
        
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' not found. Did you provide the correct path?")
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
    
    results = model.track(
        source=str(target_video),
        tracker=args.tracker,
        save=True,       
        conf=args.conf,  
        device="mps"     
    )
    
    # Cleanup temp file if we created one
    if temp_file and temp_file.exists():
        temp_file.unlink()
        
    print("\nTracking complete! Check the most recently created folder in 'runs/detect/track' for your annotated video.")

if __name__ == "__main__":
    main()

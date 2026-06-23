import argparse
import os
import cv2
import subprocess
import tempfile
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from core.image_utils import crop_with_padding

def main():
    parser = argparse.ArgumentParser(description="Extract crops of tracked fish")
    parser.add_argument("--video", required=True, type=str, help="Path to input video")
    parser.add_argument("--weights", default="runs/detect/fish_tracking_model/weights/best.pt", type=str)
    parser.add_argument("--padding", type=float, default=0.10, help="Padding percentage (e.g. 0.1 for 10%%)")
    parser.add_argument("--output_dir", type=str, default="crops", help="Base output directory")
    parser.add_argument("--duration", type=int, default=None, help="Only process the first N seconds of the video")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    video_name = video_path.stem
    output_base = Path(args.output_dir) / video_name
    
    # Handle duration cropping and unsupported formats like .MTS
    temp_file = None
    target_video = video_path
    
    # Ultralytics supported video extensions
    supported_exts = {'.mov', '.gif', '.m4v', '.mpeg', '.mpg', '.asf', '.ts', '.avi', '.wmv', '.mp4', '.mkv', '.webm'}
    needs_remux = video_path.suffix.lower() not in supported_exts
    
    if args.duration is not None or needs_remux:
        action = f"Cropping the first {args.duration} seconds" if args.duration else "Remuxing unsupported format"
        print(f"{action} using FFmpeg...")
        temp_dir = tempfile.gettempdir()
        temp_file = Path(temp_dir) / f"trimmed_{video_path.stem}.mp4"
        
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        if args.duration is not None:
            cmd.extend(["-t", str(args.duration)])
            
        # If it's a natively supported format, just copy the stream to save time.
        # If it's unsupported (like .MTS), let ffmpeg re-encode it to a standard progressive format
        # to fix the interlaced frame count metadata bug.
        if not needs_remux:
            cmd.extend(["-c", "copy"])
            
        cmd.append(str(temp_file))
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            target_video = temp_file
            print("FFmpeg processing successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error processing video with FFmpeg: {e}")
            return
    
    print(f"Loading YOLO model from {args.weights}")
    model = YOLO(args.weights)
    
    print(f"Processing video {target_video}")
    # Run tracking in stream mode for efficiency
    results = model.track(source=str(target_video), stream=True, save=False, conf=0.4, device="mps")
    
    frame_idx = 0
    saved_crops = 0
    for r in results:
        frame_idx += 1
        if r.boxes is None or r.boxes.id is None:
            continue
            
        img = r.orig_img
        h_img, w_img = img.shape[:2]
        
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        
        for box, obj_id in zip(boxes, ids):
            crop = crop_with_padding(img, box, padding=args.padding)
            
            if crop.size == 0:
                continue
                
            save_dir = output_base / f"id_{obj_id}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"frame_{frame_idx}.jpg"
            cv2.imwrite(str(save_path), crop)
            saved_crops += 1
            
    # Cleanup temp file if we created one
    if temp_file and temp_file.exists():
        temp_file.unlink()
            
    print(f"Finished extracting {saved_crops} crops to {output_base}")

if __name__ == '__main__':
    main()

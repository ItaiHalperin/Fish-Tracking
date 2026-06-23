import cv2
import random
from pathlib import Path

def find_video_files(videos_dir: Path) -> list[Path]:
    """Scans a directory for common video extensions."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm", ".mts"}
    return [
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

def gather_frame_metadata(video_files: list[Path], num_frames: int) -> list[dict]:
    """Reads video files, determines frame counts, and randomly selects frames to extract."""
    all_frames_metadata = []

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video file '{video_path}'. Skipping.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames <= 0:
            print(f"Warning: Video '{video_path}' reports 0 or negative frame count. Skipping.")
            continue

        if fps <= 0:
            fps = 30.0
            print(f"Warning: FPS is not reported for '{video_path}'. Defaulting to 30.0.")

        # Use only the first 90% of reported frames — remuxed MTS files often
        # over-report their frame count, making the tail unreadable.
        safe_frames = int(total_frames * 0.9)
        num_to_select = min(num_frames, safe_frames)
        selected_indices = random.sample(range(safe_frames), num_to_select)

        for idx in selected_indices:
            timestamp = idx / fps
            all_frames_metadata.append({
                "video_path": video_path,
                "video_name": video_path.stem,
                "frame_idx": idx,
                "timestamp": timestamp
            })

    return all_frames_metadata

def extract_and_save_frames(extraction_plan: dict, output_dir: Path) -> int:
    """Executes the extraction plan using direct seeking for robustness.
    
    Frames are organized into subfolders by video name, e.g.:
        output_dir/video1/video1_time_1.234s.jpg
        output_dir/video2/video2_time_0.500s.jpg
    """
    extracted_count = 0

    for video_path, targets in extraction_plan.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Failed to open '{video_path}' for extraction. Skipping.")
            continue

        video_name = video_path.stem

        # Create a subfolder for this video
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        video_extracted = 0
        for frame_idx, timestamp in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            time_str = f"{timestamp:.3f}s"
            filename = f"{video_name}_time_{time_str}.jpg"
            save_path = video_output_dir / filename
            cv2.imwrite(str(save_path), frame)
            video_extracted += 1
            extracted_count += 1

        cap.release()
        print(f"  Processed '{video_name}': extracted {video_extracted}/{len(targets)} frames → {video_output_dir}")

    return extracted_count

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

        num_to_select = min(num_frames, total_frames)
        selected_indices = random.sample(range(total_frames), num_to_select)

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
    """Executes the extraction plan, seeking frames efficiently and saving them."""
    extracted_count = 0

    for video_path, targets in extraction_plan.items():
        targets.sort(key=lambda x: x[0])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Failed to open '{video_path}' for extraction. Skipping.")
            continue

        video_name = video_path.stem

        target_idx_pointer = 0
        current_frame = 0

        while target_idx_pointer < len(targets):
            ret = cap.grab()
            if not ret:
                print(f"Warning: Reached end of video or failed to grab frame {current_frame} in '{video_name}'.")
                break
                
            expected_idx, timestamp = targets[target_idx_pointer]
            
            if current_frame == expected_idx:
                ret, frame = cap.retrieve()
                if not ret:
                    print(f"Warning: Failed to retrieve frame {current_frame} from '{video_name}'. Skipping.")
                else:
                    time_str = f"{timestamp:.3f}s"
                    filename = f"{video_name}_time_{time_str}.jpg"
                    # Just dump the image straight into the target folder
                    save_path = output_dir / filename
                    cv2.imwrite(str(save_path), frame)
                    extracted_count += 1
                
                target_idx_pointer += 1
            
            current_frame += 1

        cap.release()
        print(f"Processed '{video_name}': extracted {len(targets)} frames.")

    return extracted_count

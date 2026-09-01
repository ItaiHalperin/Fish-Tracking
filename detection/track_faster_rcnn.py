#!/usr/bin/env python3
"""
Faster R-CNN Tracking Script for FishTracking.

Runs the trained PyTorch Faster R-CNN model on a video to detect and track 
goldfish across frames using ByteTrack (via the `trackers` library), 
replicating the output format of Ultralytics YOLO.
"""

import sys
import yaml
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.device import get_device

try:
    import supervision as sv
except ImportError:
    print("Error: The 'supervision' package is required for Faster R-CNN tracking.")
    print("Please install it by running: .venv/bin/pip install supervision")
    sys.exit(1)

try:
    from trackers import ByteTrackTracker
except ImportError:
    print("Error: The 'trackers' package is required for ByteTrack tracking.")
    print("Please install it by running: .venv/bin/pip install trackers")
    sys.exit(1)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.ml_storage import MLStorage

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

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
    parser = argparse.ArgumentParser(description="Track goldfish using Faster R-CNN and ByteTrack.")
    parser.add_argument("--video", required=True, type=str, help="Path to the input video (.mp4, .MTS, etc.)")
    parser.add_argument("--output", required=True, type=str, help="Output video file path for tracking results (e.g., output/tracked.mp4)")
    parser.add_argument("--weights", default=None,
                        type=str, help="Path to the trained best.pt weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Only process N seconds of the video (from --start)")
    # ML Storage options
    parser.add_argument("--use-storage", action="store_true",
                        help="Fetch weights from the MLStorage model registry instead of --weights")
    parser.add_argument("--storage-root", type=str, default="ml_storage", help="Root directory for MLStorage")
    parser.add_argument("--model-name", type=str, default="goldfish_faster_rcnn", help="Model name in registry")
    parser.add_argument("--run", type=int, default=None, help="Run number from registry (default: latest)")
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

    # Parse dataset config to get number of classes and names
    with open("dataset.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    # The model was trained with 2 classes (0: background, 1: goldfish) to fix the phantom class issue
    num_classes = 2
    class_names = data_config["names"] # {0: 'name', 1: 'name'}
    
    device = torch.device(get_device())
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading Faster R-CNN model from {weights_path}...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Setup ByteTrack tracker and Supervision annotators
    byte_tracker = ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(str(target_video))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting tracking on {target_video.name} using ByteTrack (conf={args.conf})...")

    with sv.VideoSink(args.output, video_info=video_info) as sink:
        # Loop over frames using supervision's generator
        for frame in sv.get_video_frames_generator(str(target_video)):
            # Faster R-CNN expects RGB tensor [0, 1]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = torchvision.transforms.functional.to_tensor(rgb_frame).to(device)

            with torch.no_grad():
                # Model expects a list of tensors
                outputs = model([tensor_frame])[0]

            # Move outputs to CPU and convert to numpy
            boxes = outputs["boxes"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            # The model outputs class 1 for goldfish. Since YOLO class 1 is also goldfish,
            # we don't need to shift it (Faster R-CNN filters out background class 0).
            class_ids = outputs["labels"].cpu().numpy() 

            # Filter by confidence
            mask = scores >= args.conf
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if len(boxes) > 0:
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=scores,
                    class_id=class_ids
                )
            else:
                detections = sv.Detections.empty()

            # Always update ByteTrack so its internal frame counter advances
            detections = byte_tracker.update(detections)

            if len(detections) > 0:
                # Annotate Frame
                labels = [
                    f"#{tracker_id} {class_names.get(class_id, 'unknown')} {confidence:.2f}"
                    for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)
                ]
                
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            else:
                annotated_frame = frame

            # Write to output
            sink.write_frame(annotated_frame)

    # Cleanup temp file if we created one
    if temp_file and temp_file.exists():
        temp_file.unlink()

    print(f"\nTracking complete! Video saved to '{output_path}'.")

if __name__ == "__main__":
    main()

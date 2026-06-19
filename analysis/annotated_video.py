#!/usr/bin/env python3
"""
AnnotatedVideoRenderer — fused detection + classification video output.

Runs the YOLO detector / ByteTrack tracker and the YOLO classifier
simultaneously, frame-by-frame, and writes an annotated video that shows:

    #<id>  <det_conf>  |  <cls_label> <cls_conf>

next to each bounding box.

Temporal smoothing (majority-vote sliding window) is applied per-track ID,
so the displayed class is stable rather than flickering.

Weights can be supplied as direct paths or resolved from the MLStorage
model registry (--use-storage).

Usage (CLI)
-----------
    python -m analysis.annotated_video \\
        --video videos/tank_A.MTS \\
        --output runs/annotated/tank_A \\
        --use-storage

Usage (API)
-----------
    from analysis.annotated_video import AnnotatedVideoRenderer
    from core.ml_storage import MLStorage

    renderer = AnnotatedVideoRenderer(storage=MLStorage("ml_storage"))
    renderer.render("videos/tank_A.MTS", output_path="runs/annotated/tank_A.mp4")
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
import tempfile
from collections import deque, Counter
from typing import Deque

import cv2
import numpy as np
from ultralytics import YOLO

from core.image_utils import crop_with_padding
from core.ml_storage import MLStorage

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DETECTION_MODEL  = "goldfish_yolo"
DEFAULT_CLASSIFIER_MODEL = "fish_position_classifier"

# Colours (BGR)
_BOX_COLOUR    = (0, 200, 255)   # amber-yellow  — bounding box & ID chip
_LABEL_COLOUR  = (50, 230, 120)  # mint green    — classifier chip
_TEXT_COLOUR   = (15,  15,  15)  # near-black    — text on chips
_UNKNOWN_COLOUR = (120, 120, 120) # grey          — when class not yet known

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.52
_THICKNESS  = 1
_PAD        = 4   # pixels of padding inside label chips


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_video(
    video_path: Path,
    start: int = 0,
    duration: int | None = None,
) -> tuple[Path, Path | None]:
    """Return (target_video, temp_file_or_None).

    Remuxes unsupported formats and/or trims to [start, start+duration].
    Caller must delete temp_file when done.
    """
    _SUPPORTED = {".mov", ".gif", ".m4v", ".mpeg", ".mpg", ".asf", ".ts",
                  ".avi", ".wmv", ".mp4", ".mkv", ".webm"}
    needs_remux = video_path.suffix.lower() not in _SUPPORTED
    needs_trim  = duration is not None or start > 0

    if not needs_remux and not needs_trim:
        return video_path, None

    tmp = Path(tempfile.gettempdir()) / f"avr_{video_path.stem}.mp4"
    cmd = ["ffmpeg", "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video_path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    if not needs_remux:
        cmd += ["-c", "copy"]
    cmd.append(str(tmp))

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp, tmp


def _draw_chip(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg: tuple[int, int, int],
    anchor: str = "top-left",
) -> tuple[int, int, int, int]:
    """Draw a filled rounded rectangle with *text* anchored at (x, y).

    anchor: 'top-left' | 'top-right'
    Returns (x1, y1, x2, y2) of the drawn chip.
    """
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    chip_w = tw + 2 * _PAD
    chip_h = th + baseline + 2 * _PAD

    if anchor == "top-right":
        x1, y1 = x - chip_w, y
    else:
        x1, y1 = x, y
    x2, y2 = x1 + chip_w, y1 + chip_h

    # Clip to frame bounds
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_frame, x2); y2 = min(h_frame, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, cv2.FILLED)
    cv2.putText(
        frame, text,
        (x1 + _PAD, y2 - _PAD - baseline),
        _FONT, _FONT_SCALE, _TEXT_COLOUR, _THICKNESS, cv2.LINE_AA,
    )
    return x1, y1, x2, y2


def _annotate_frame(
    frame: np.ndarray,
    boxes:    np.ndarray,    # (N, 4)  xyxy
    track_ids: np.ndarray,   # (N,)    int
    det_confs: np.ndarray,   # (N,)    float
    cls_labels: list[str],   # (N,)    str | "?"
    cls_confs:  list[float], # (N,)    float | None
) -> np.ndarray:
    """Draw all annotations onto a copy of *frame* and return it."""
    out = frame.copy()

    for box, tid, det_conf, cls_label, cls_conf in zip(
        boxes, track_ids, det_confs, cls_labels, cls_confs
    ):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOUR, 2)

        # ── Left chip: ID + detection confidence ──────────────────────────
        id_text = f"#{tid}  {det_conf:.2f}"
        chip_x1, chip_y1, chip_x2, chip_y2 = _draw_chip(
            out, id_text, x1, max(0, y1 - (_FONT_SCALE * 30 + 2 * _PAD + 2).__ceil__()),
            _BOX_COLOUR, anchor="top-left",
        )

        # ── Right chip: classifier label + confidence ──────────────────────
        if cls_label == "?":
            cls_text = "?"
            bg = _UNKNOWN_COLOUR
        else:
            cls_text = f"{cls_label}  {cls_conf:.2f}" if cls_conf is not None else cls_label
            bg = _LABEL_COLOUR

        # Place the classifier chip just to the right of the ID chip (same row)
        _draw_chip(out, cls_text, chip_x2 + 2, chip_y1, bg, anchor="top-left")

    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AnnotatedVideoRenderer:
    """Render a tracking + classification video from a single source video.

    Parameters
    ----------
    storage:
        ``MLStorage`` instance used to resolve model weights.
        Pass ``None`` only when supplying explicit weight paths.
    detection_weights:
        Direct path to detector ``best.pt``.  Takes precedence over *storage*.
    classifier_weights:
        Direct path to classifier ``best.pt``.  Takes precedence over *storage*.
    detection_model_name:
        Registry key for the detector (default: ``"goldfish_yolo"``).
    classifier_model_name:
        Registry key for the classifier (default: ``"fish_position_classifier"``).
    conf:
        Detection confidence threshold (default: ``0.4``).
    tracker:
        Tracker config file (default: ``"bytetrack.yaml"``).
    padding:
        Fractional padding added around each bounding box before cropping for
        classification (default: ``0.10`` = 10 %).  Must match the value used
        by ``extract_crops.py`` when the training data was generated.
    smooth_window:
        Sliding-window size for temporal classification smoothing (default: ``5``).
    reject_class:
        Class whose frames are counted as "rejected" in the label (default: ``"unclear"``).
    device:
        Torch device string (default: ``"mps"``).
    """

    def __init__(
        self,
        storage: MLStorage | None = None,
        *,
        detection_weights:    Path | str | None = None,
        classifier_weights:   Path | str | None = None,
        detection_model_name:  str = DEFAULT_DETECTION_MODEL,
        classifier_model_name: str = DEFAULT_CLASSIFIER_MODEL,
        conf:          float = 0.4,
        tracker:       str   = "bytetrack.yaml",
        padding:       float = 0.10,
        smooth_window: int   = 5,
        reject_class:  str   = "unclear",
        device:        str   = "mps",
    ):
        self.storage                = storage
        self._det_weights_override  = Path(detection_weights)  if detection_weights  else None
        self._cls_weights_override  = Path(classifier_weights) if classifier_weights else None
        self.detection_model_name   = detection_model_name
        self.classifier_model_name  = classifier_model_name
        self.conf          = conf
        self.tracker       = tracker
        self._pad          = padding   # matches extract_crops.py default of 0.10
        self.smooth_window = smooth_window
        self.reject_class  = reject_class
        self.device        = device

        # Lazy-loaded
        self._detector:   YOLO | None = None
        self._classifier: YOLO | None = None

    # ------------------------------------------------------------------
    # Weight resolution (lazy)
    # ------------------------------------------------------------------

    def _get_detector(self) -> YOLO:
        if self._detector is None:
            if self._det_weights_override:
                path = self._det_weights_override
            elif self.storage:
                path = self.storage.detection_models.get_weights(self.detection_model_name)
            else:
                raise ValueError("No detector weights: pass detection_weights= or a storage=.")
            print(f"[AnnotatedVideoRenderer] Loading detector  → {path}")
            self._detector = YOLO(path)
        return self._detector

    def _get_classifier(self) -> YOLO:
        if self._classifier is None:
            if self._cls_weights_override:
                path = self._cls_weights_override
            elif self.storage:
                path = self.storage.classifier_models.get_weights(self.classifier_model_name)
            else:
                raise ValueError("No classifier weights: pass classifier_weights= or a storage=.")
            print(f"[AnnotatedVideoRenderer] Loading classifier → {path}")
            self._classifier = YOLO(path)
        return self._classifier

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def render(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        start:    int       = 0,
        duration: int | None = None,
    ) -> Path:
        """Run the fused detector + classifier and write an annotated video.

        Parameters
        ----------
        video_path:
            Input video (any format supported by FFmpeg / OpenCV).
        output_path:
            Destination ``.mp4`` file (parent directories are created).
        start:
            Start offset in seconds (default: ``0``).
        duration:
            Clip length in seconds; ``None`` processes the whole video.

        Returns
        -------
        Path
            Absolute path to the written output file.
        """
        video_path  = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")

        # Prepare (remux / trim if needed)
        target_video, temp_file = _prepare_video(video_path, start, duration)

        try:
            detector   = self._get_detector()
            classifier = self._get_classifier()

            # Open the source to read fps / frame size
            cap = cv2.VideoCapture(str(target_video))
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {target_video}")
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            # Per-track sliding window buffers  {track_id: deque[str]}
            label_buffers: dict[int, Deque[str]] = {}

            print(
                f"[AnnotatedVideoRenderer] Rendering {video_path.name} "
                f"({total} frames @ {fps:.1f} fps) → {output_path.name}"
            )

            frame_idx = 0
            for result in detector.track(
                source=str(target_video),
                tracker=self.tracker,
                stream=True,
                conf=self.conf,
                device=self.device,
            ):
                frame_idx += 1
                frame = result.orig_img

                if result.boxes is None or result.boxes.id is None:
                    # No detections this frame — write the bare frame
                    writer.write(frame)
                    if frame_idx % 100 == 0:
                        print(f"  frame {frame_idx}/{total}")
                    continue

                boxes     = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                det_confs = result.boxes.conf.cpu().numpy()

                # --- Classify each crop on this frame ------------------
                # Apply the same 10% padding used by extract_crops.py so
                # the crops match the training distribution exactly.
                crops = []
                for box in boxes:
                    crop = crop_with_padding(frame, box, padding=self._pad, min_size=(32, 32))
                    crops.append(crop)

                cls_results = classifier.predict(
                    source=crops,
                    device=self.device,
                    verbose=False,
                )

                cls_labels: list[str]        = []
                cls_confs:  list[float | None] = []
                for tid, res in zip(track_ids, cls_results):
                    raw_label = classifier.names[int(res.probs.top1)]
                    raw_conf  = float(res.probs.top1conf)

                    # Update sliding window for this track
                    buf = label_buffers.setdefault(
                        tid, deque(maxlen=self.smooth_window)
                    )
                    buf.append(raw_label)

                    smoothed = Counter(buf).most_common(1)[0][0]
                    cls_labels.append(smoothed)
                    # Show conf of the smoothed class (if it matches raw)
                    cls_confs.append(raw_conf if raw_label == smoothed else None)

                annotated = _annotate_frame(
                    frame, boxes, track_ids, det_confs, cls_labels, cls_confs
                )
                writer.write(annotated)

                if frame_idx % 100 == 0:
                    print(f"  frame {frame_idx}/{total}")

            writer.release()

        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()

        print(f"[AnnotatedVideoRenderer] Done → {output_path.resolve()}")
        return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a fused detection + classification annotated video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    parser.add_argument("--video",  required=True,  help="Input video path")
    parser.add_argument("--output", required=True,  help="Output .mp4 path")
    parser.add_argument("--start",    type=int,   default=0,    help="Start offset in seconds")
    parser.add_argument("--duration", type=int,   default=None, help="Clip length in seconds")

    # Tracker / detector
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Detection confidence threshold")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help="Tracking algorithm")
    parser.add_argument("--device",  default="mps",
                        help="Torch device (mps / cuda / cpu)")

    # Classifier
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Temporal smoothing window for classification")
    parser.add_argument("--reject-class",  default="unclear",
                        help="Class name excluded from scored predictions")
    parser.add_argument("--padding", type=float, default=0.10,
                        help="Fractional padding around each box before classifying "
                             "(default 0.10 — must match the value used during crop extraction)")

    # Weight resolution — direct paths
    parser.add_argument("--det-weights", default=None,
                        help="Direct path to detector best.pt")
    parser.add_argument("--cls-weights", default=None,
                        help="Direct path to classifier best.pt")

    # Weight resolution — ML Storage
    parser.add_argument("--use-storage", action="store_true",
                        help="Resolve weights from MLStorage registry")
    parser.add_argument("--storage-root",  default="ml_storage",
                        help="MLStorage root directory")
    parser.add_argument("--det-model-name", default=DEFAULT_DETECTION_MODEL,
                        help="Detector model name in the registry")
    parser.add_argument("--cls-model-name", default=DEFAULT_CLASSIFIER_MODEL,
                        help="Classifier model name in the registry")
    parser.add_argument("--det-run", type=int, default=None,
                        help="1-indexed detector run (default: latest)")
    parser.add_argument("--cls-run", type=int, default=None,
                        help="1-indexed classifier run (default: latest)")

    args = parser.parse_args()

    # --- Resolve weights ------------------------------------------------
    det_weights: Path | None = None
    cls_weights: Path | None = None
    storage: MLStorage | None = None

    if args.det_weights:
        det_weights = Path(args.det_weights)
        if not det_weights.exists():
            parser.error(f"--det-weights not found: {det_weights}")

    if args.cls_weights:
        cls_weights = Path(args.cls_weights)
        if not cls_weights.exists():
            parser.error(f"--cls-weights not found: {cls_weights}")

    if args.use_storage or (det_weights is None or cls_weights is None):
        storage = MLStorage(args.storage_root)
        if det_weights is None:
            det_weights = storage.detection_models.get_weights(
                args.det_model_name, run=args.det_run
            )
        if cls_weights is None:
            cls_weights = storage.classifier_models.get_weights(
                args.cls_model_name, run=args.cls_run
            )

    # --- Build renderer and run -----------------------------------------
    renderer = AnnotatedVideoRenderer(
        storage=storage,
        detection_weights=det_weights,
        classifier_weights=cls_weights,
        detection_model_name=args.det_model_name,
        classifier_model_name=args.cls_model_name,
        conf=args.conf,
        tracker=args.tracker,
        padding=args.padding,
        smooth_window=args.smooth_window,
        reject_class=args.reject_class,
        device=args.device,
    )

    renderer.render(
        video_path=args.video,
        output_path=args.output,
        start=args.start,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()

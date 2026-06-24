#!/usr/bin/env python3
"""
FishPipeline — end-to-end analysis of a single video.

Steps
-----
1. **Track** — run the detection model on the video; produce a tracked video
   and per-frame bounding boxes with fish IDs.
2. **Crop** — extract a padded JPEG crop for every (fish_id, frame) pair.
3. **Classify** — run the classifier on each fish's crop sequence and apply
   temporal smoothing.
4. **Summarise** — return two DataFrames:
    - ``traces``  : one row per frame_idx, one column per fish_id containing
                    the smoothed classification label.
    - ``summary`` : one row per fish_id + an AVERAGE row, columns = fish_id,
                   n_frames, n_scored, n_rejected, <class> …

Usage
-----
    from analysis.pipeline import FishPipeline
    from core.ml_storage import MLStorage

    storage = MLStorage("ml_storage")
    pipe = FishPipeline(storage=storage)
    traces, summary = pipe.run("videos/tank_A.MTS", crops_dir="crops/tank_A")
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import subprocess
import tempfile
from collections import Counter

import cv2
import pandas as pd
from ultralytics import YOLO

from core.image_utils import crop_with_padding
from core.ml_storage import MLStorage
from classifier import load_classifier

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DETECTION_MODEL  = "goldfish_yolo"
DEFAULT_CLASSIFIER_MODEL = "fish_position_classifier"
_SUPPORTED_EXTS = {".mov", ".gif", ".m4v", ".mpeg", ".mpg", ".asf", ".ts",
                   ".avi", ".wmv", ".mp4", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Small helpers (re-implemented here so the pipeline is self-contained)
# ---------------------------------------------------------------------------

def _smooth(labels: list[str], window: int) -> list[str]:
    """Majority-vote temporal smoothing over a sliding window."""
    if window <= 1 or len(labels) <= 1:
        return labels
    half = window // 2
    out = []
    for i in range(len(labels)):
        chunk = labels[max(0, i - half): i + half + 1]
        out.append(Counter(chunk).most_common(1)[0][0])
    return out


def _frame_index(path: Path) -> int:
    """Parse frame index from filenames like ``frame_123.jpg``."""
    return int(path.stem.split("_")[-1])


def _prepare_video(video_path: Path, start: int = 0, duration: int | None = None) -> tuple[Path, Path | None]:
    """Return (target_video, temp_file).

    Remuxes unsupported formats and/or trims to [start, start+duration].
    Caller is responsible for deleting temp_file when done.
    """
    needs_remux = video_path.suffix.lower() not in _SUPPORTED_EXTS
    needs_trim  = duration is not None or start > 0

    if not needs_remux and not needs_trim:
        return video_path, None

    temp_file = Path(tempfile.gettempdir()) / f"pipeline_{video_path.stem}.mp4"
    cmd = ["ffmpeg", "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video_path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    if not needs_remux:
        cmd += ["-c", "copy"]
    cmd.append(str(temp_file))

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_file, temp_file


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FishPipeline:
    """Run detection → crop extraction → classification on one video.

    Parameters
    ----------
    storage:
        An ``MLStorage`` instance.  Weights are resolved automatically:
        ``default.pt`` if present, otherwise the latest trained run.
    detection_model_name:
        Key in ``storage.detection_models`` (default: ``"goldfish_yolo"``).
    classifier_model_name:
        Key in ``storage.classifier_models``
        (default: ``"fish_position_classifier"``).
    conf:
        Detection confidence threshold (default ``0.4``).
    padding:
        Fractional padding added around each bounding box before cropping
        (default ``0.10`` = 10 %).
    smooth_window:
        Temporal smoothing window for classification labels (default ``5``).
    reject_class:
        Class name excluded from scored percentages (default ``"unclear"``).
    device:
        Torch device string (default ``"mps"``).
    tracker:
        ByteTrack or BoTSORT config file (default ``"bytetrack.yaml"``).
    """

    def __init__(
        self,
        storage: MLStorage,
        detection_model_name:  str = DEFAULT_DETECTION_MODEL,
        classifier_model_name: str = DEFAULT_CLASSIFIER_MODEL,
        conf:          float = 0.4,
        padding:       float = 0.10,
        smooth_window: int   = 5,
        reject_class:  str   = "unclear",
        device:        str   = "mps",
        tracker:       str   = "bytetrack.yaml",
    ):
        self.storage               = storage
        self.detection_model_name  = detection_model_name
        self.classifier_model_name = classifier_model_name
        self.conf          = conf
        self.padding       = padding
        self.smooth_window = smooth_window
        self.reject_class  = reject_class
        self.device        = device
        self.tracker       = tracker

        # Lazy-loaded; reused across multiple .run() calls
        self._detector:   YOLO | None = None
        self._classifier = None  # a classifier.Classifier, resolved from storage

    # ------------------------------------------------------------------
    # Weight resolution
    # ------------------------------------------------------------------

    def _get_detector(self) -> YOLO:
        if self._detector is None:
            weights = self.storage.detection_models.get_weights(self.detection_model_name)
            print(f"[Pipeline] Loading detector from {weights}")
            self._detector = YOLO(weights)
        return self._detector

    def _get_classifier(self):
        if self._classifier is None:
            weights = self.storage.classifier_models.get_weights(self.classifier_model_name)
            print(f"[Pipeline] Loading classifier from {weights}")
            # Resolves to our Classifier (multihead / roll_cls / angle_reg) via the
            # run's config.yaml, or a YOLO adapter for legacy weights.
            self._classifier = load_classifier(weights, device=self.device)
        return self._classifier

    # ------------------------------------------------------------------
    # Steps 1+2: Track and extract crops (streamed)
    # ------------------------------------------------------------------

    def _track_and_extract(
        self,
        video_path: Path,
        crops_dir: Path,
        track_output_dir: Path | None,
        start: int,
        duration: int | None,
    ) -> None:
        """Track in stream mode and save each frame's crops as it arrives.

        Crops are written per frame so only one Result is ever held in memory —
        materialising every frame (``list(model.track(...))``) OOM-kills on long
        videos because each Result keeps its full image.
        """
        target_video, temp_file = _prepare_video(video_path, start, duration)
        try:
            model = self._get_detector()
            print(f"[Pipeline] Tracking {target_video.name} (streaming → {crops_dir}) …")

            kwargs = dict(
                source=str(target_video),
                tracker=self.tracker,
                stream=True,
                conf=self.conf,
                device=self.device,
                save=track_output_dir is not None,
            )
            if track_output_dir is not None:
                kwargs["project"] = str(track_output_dir.parent)
                kwargs["name"]    = track_output_dir.name

            saved = 0
            for frame_idx, r in enumerate(model.track(**kwargs), start=1):
                if r.boxes is None or r.boxes.id is None:
                    continue

                img   = r.orig_img
                boxes = r.boxes.xyxy.cpu().numpy()
                ids   = r.boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    crop = crop_with_padding(img, box, padding=self.padding)
                    if crop.size == 0:
                        continue
                    fish_dir = crops_dir / f"id_{obj_id}"
                    fish_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(fish_dir / f"frame_{frame_idx}.jpg"), crop)
                    saved += 1

                if frame_idx % 1000 == 0:
                    print(f"  frame {frame_idx}: {saved} crops saved so far")
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()

        print(f"[Pipeline] Saved {saved} crops.")

    # ------------------------------------------------------------------
    # Step 3 + 4: Classify and build DataFrames
    # ------------------------------------------------------------------

    def _classify_and_summarise(
        self, crops_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (traces_df, summary_df)."""
        fish_dirs = sorted(
            [p for p in crops_dir.iterdir() if p.is_dir() and p.name.startswith("id_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not fish_dirs:
            raise RuntimeError(f"No id_* subfolders found in {crops_dir}")

        model      = self._get_classifier()
        class_names     = model.class_names
        reported_classes = [c for c in class_names if c != self.reject_class]

        print(f"[Pipeline] Classifying {len(fish_dirs)} fish IDs …")

        trace_rows   = []
        summary_rows = []

        for fish_dir in fish_dirs:
            fish_id = fish_dir.name          # e.g. "id_3"
            crops   = sorted(fish_dir.glob("frame_*.jpg"), key=_frame_index)
            if not crops:
                continue

            preds = model.predict([str(c) for c in crops])
            raw_labels = [label for label, _conf in preds]
            smoothed   = _smooth(raw_labels, self.smooth_window)

            # Trace: one row per frame
            for crop_path, raw, smo in zip(crops, raw_labels, smoothed):
                trace_rows.append({
                    "fish_id":        fish_id,
                    "frame_idx":      _frame_index(crop_path),
                    "raw_label":      raw,
                    "smoothed_label": smo,
                })

            # Summary: per-fish counts → percentages
            counts = Counter(smoothed)
            scored = sum(counts.get(c, 0) for c in reported_classes)
            if scored == 0:
                continue

            row = {
                "fish_id":    fish_id,
                "n_frames":   len(smoothed),
                "n_scored":   scored,
                "n_rejected": counts.get(self.reject_class, 0),
            }
            for cls in reported_classes:
                row[cls] = 100.0 * counts.get(cls, 0) / scored
            summary_rows.append(row)

        traces_long = pd.DataFrame(trace_rows)
        if not traces_long.empty:
            traces_df = traces_long.pivot(
                index="frame_idx", columns="fish_id", values="smoothed_label",
            )
            traces_df.index.name = "frame_idx"
            traces_df.columns.name = None          # drop the "fish_id" axis label
            traces_df = traces_df.sort_index()
        else:
            traces_df = pd.DataFrame()
        summary_df = pd.DataFrame(summary_rows)

        if not summary_df.empty:
            avg = {
                "fish_id":    "AVERAGE",
                "n_frames":   summary_df["n_frames"].sum(),
                "n_scored":   summary_df["n_scored"].sum(),
                "n_rejected": summary_df["n_rejected"].sum(),
            }
            total_scored = avg["n_scored"]
            for cls in reported_classes:
                # Weighted average: convert each fish's % back to counts,
                # sum across fish, then divide by the overall scored total.
                avg[cls] = (summary_df[cls] * summary_df["n_scored"]).sum() / (100.0 * total_scored) * 100.0 if total_scored else 0.0
            summary_df = pd.concat(
                [summary_df, pd.DataFrame([avg])], ignore_index=True
            )

        return traces_df, summary_df

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str | Path,
        *,
        crops_dir:       str | Path | None = None,
        track_output_dir: str | Path | None = None,
        start:    int       = 0,
        duration: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full pipeline on *video_path*.

        Parameters
        ----------
        video_path:
            Input video file.
        crops_dir:
            Where to save per-fish crop images.  Defaults to
            ``crops/<video_stem>/`` relative to the current working directory.
        track_output_dir:
            Where to save the annotated tracking video.  ``None`` (default)
            skips saving the video.
        start:
            Start offset in seconds (default ``0``).
        duration:
            Clip length in seconds; ``None`` processes the whole video.

        Returns
        -------
        traces : pd.DataFrame
            Per-frame smoothed labels for every fish.
            Index: ``frame_idx``; one column per ``fish_id``.
        summary : pd.DataFrame
            Per-fish position percentages + AVERAGE row.
            Columns: ``fish_id``, ``n_frames``, ``n_scored``, ``n_rejected``,
            ``<class_1>``, …, ``<class_N>``.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if crops_dir is None:
            crops_dir = Path("crops") / video_path.stem
        crops_dir = Path(crops_dir)

        if track_output_dir is not None:
            track_output_dir = Path(track_output_dir)

        print(f"\n[Pipeline] ── {video_path.name} ──────────────────────────")

        # 1+2. Track and extract crops (streamed, memory-flat)
        self._track_and_extract(video_path, crops_dir, track_output_dir, start, duration)

        # 3+4. Classify and summarise
        traces_df, summary_df = self._classify_and_summarise(crops_dir)

        print(f"[Pipeline] Done.  {len(summary_df) - 1} fish summarised.\n")
        return traces_df, summary_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the FishPipeline on a video and save results as spreadsheets.",
    )
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Directory where traces.xlsx and summary.xlsx will be saved.",
    )
    parser.add_argument("--crops-dir", type=str, default=None,
                        help="Directory for per-fish crop images (default: crops/<video_stem>).")
    parser.add_argument("--track-output-dir", type=str, default=None,
                        help="Directory to save the annotated tracking video.")
    parser.add_argument("--storage", type=str, default="ml_storage",
                        help="Path to the MLStorage directory (default: ml_storage).")
    parser.add_argument("--start", type=int, default=0,
                        help="Start offset in seconds (default: 0).")
    parser.add_argument("--duration", type=int, default=None,
                        help="Clip length in seconds (default: full video).")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4).")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Temporal smoothing window (default: 5).")
    parser.add_argument("--device", type=str, default="mps",
                        help="Torch device (default: mps).")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = MLStorage(args.storage)
    pipe = FishPipeline(
        storage=storage,
        conf=args.conf,
        smooth_window=args.smooth_window,
        device=args.device,
    )

    traces_df, summary_df = pipe.run(
        args.video,
        crops_dir=args.crops_dir,
        track_output_dir=args.track_output_dir,
        start=args.start,
        duration=args.duration,
    )

    traces_path  = output_dir / "traces.xlsx"
    summary_path = output_dir / "summary.xlsx"

    traces_df.to_excel(traces_path)
    summary_df.to_excel(summary_path, index=False)

    print(f"[Pipeline] Saved {traces_path}")
    print(f"[Pipeline] Saved {summary_path}")

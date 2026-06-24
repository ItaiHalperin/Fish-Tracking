#!/usr/bin/env python3
"""
Fish analysis webapp.

Drag in a video (or paste a local path), and it runs the full pipeline —
detection + tracking → per-fish crops → roll classification → summary — and
shows the per-fish + pooled "average over all crops" breakdown in a clean page.

The job runs in a background thread with live progress (the pipeline's per-1000-
frame log is streamed to the page). Crops go to a temp dir per run, so nothing
is left behind and stale crops can't contaminate results.

Usage:
    python webapp/analyze_app.py            # http://127.0.0.1:5070
    python webapp/analyze_app.py --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, render_template, request

from analysis.pipeline import FishPipeline
from core.ml_storage import MLStorage

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3  # 4 GB uploads

DEVICE = "mps"
_jobs: dict[str, dict] = {}
_lock = threading.Lock()          # one analysis at a time (single GPU)
_BASE_COLS = ("fish_id", "n_frames", "n_scored", "n_rejected")


class _Tee(io.TextIOBase):
    """Mirror pipeline stdout into a job's log list (and the real console)."""

    def __init__(self, log: list[str]):
        self._log = log

    def write(self, s: str) -> int:
        for line in s.splitlines():
            if line.strip():
                self._log.append(line.rstrip())
        sys.__stdout__.write(s)
        return len(s)


def _summarise(summary_df) -> dict:
    if summary_df is None or summary_df.empty:
        return {"class_cols": [], "rows": [], "average": None, "n_fish": 0}
    class_cols = [c for c in summary_df.columns if c not in _BASE_COLS]
    rows = summary_df.to_dict("records")
    for r in rows:
        for c in class_cols:
            r[c] = round(float(r[c]), 1)
    average = next((r for r in rows if r["fish_id"] == "AVERAGE"), None)
    fish_rows = [r for r in rows if r["fish_id"] != "AVERAGE"]
    return {
        "class_cols": class_cols,
        "rows": fish_rows,
        "average": average,
        "n_fish": len(fish_rows),
        "total_frames": int(summary_df.loc[summary_df["fish_id"] == "AVERAGE",
                                           "n_frames"].iloc[0]) if average else 0,
    }


def _run_job(job_id: str, video_path: Path, duration: int | None, cleanup: Path | None):
    job = _jobs[job_id]
    crops_dir = Path(tempfile.mkdtemp(prefix="fishpipe_crops_"))
    try:
        with _lock:
            job["status"] = "running"
            with contextlib.redirect_stdout(_Tee(job["log"])):
                storage = MLStorage(str(REPO_ROOT / "ml_storage"))
                pipe = FishPipeline(storage=storage, device=DEVICE)
                _traces, summary = pipe.run(video_path, crops_dir=crops_dir,
                                            duration=duration)
            job["result"] = _summarise(summary)
            job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["log"].append("ERROR: " + "".join(traceback.format_exc()))
    finally:
        shutil.rmtree(crops_dir, ignore_errors=True)
        if cleanup and cleanup.exists():
            cleanup.unlink()


@app.route("/")
def index():
    return render_template("analyze.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if any(j["status"] == "running" for j in _jobs.values()):
        return jsonify({"error": "An analysis is already running. Wait for it to finish."}), 409

    duration = request.form.get("duration", type=int) or None
    upload = request.files.get("video")
    cleanup = None

    if upload and upload.filename:
        suffix = Path(upload.filename).suffix or ".mp4"
        tmp = Path(tempfile.mkdtemp(prefix="fishpipe_vid_")) / f"input{suffix}"
        upload.save(str(tmp))
        video_path = tmp
        cleanup = tmp
        source_name = upload.filename
    else:
        raw = (request.form.get("path") or "").strip().strip('"').strip("'")
        if not raw:
            return jsonify({"error": "Provide a video file or a local path."}), 400
        video_path = Path(raw).expanduser()
        if not video_path.is_absolute():
            video_path = (REPO_ROOT / video_path)
        if not video_path.exists():
            return jsonify({"error": f"Path not found: {video_path}"}), 400
        source_name = video_path.name

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "log": [], "result": None,
                     "error": None, "source": source_name}
    threading.Thread(target=_run_job, args=(job_id, video_path, duration, cleanup),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "source": source_name})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "source": job["source"],
        "progress": job["log"][-1] if job["log"] else "",
        "error": job["error"],
        "result": job["result"],
    })


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", help="Torch device (mps/cpu/cuda)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5070)
    args = ap.parse_args()
    DEVICE = args.device
    print(f"Fish analysis webapp → http://{args.host}:{args.port}  (device={DEVICE})")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

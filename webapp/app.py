#!/usr/bin/env python3
"""
Fish crop labeling web app.

Source crops live in --source (typically a *_flat folder from flatten_crops.py).
Each labeled crop is MOVED into labels_raw/<class>/. When labeling is done,
the /finish page lets you cap the per-class count and split into labels/train
and labels/val. Overflow gets parked in labels_discarded/<class>/.
"""

import argparse
import random
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# Populated in main()
SOURCE: Path = Path()
RAW: Path = Path()
OUT: Path = Path()
DISCARDED: Path = Path()
CLASSES: list[str] = []
# In-memory undo stack: (filename, class_it_was_moved_to)
UNDO: list[tuple[str, str]] = []


def list_unlabeled() -> list[str]:
    return sorted(p.name for p in SOURCE.iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def class_counts() -> dict[str, int]:
    return {c: sum(1 for _ in (RAW / c).glob("*")) for c in CLASSES}


@app.route("/")
def index():
    return render_template("label.html", classes=CLASSES)


@app.route("/finish")
def finish():
    return render_template("finish.html", classes=CLASSES)


@app.route("/api/queue")
def api_queue():
    """Return up to N unlabeled filenames for the client to work through."""
    n = int(request.args.get("n", 50))
    files = list_unlabeled()[:n]
    return jsonify({"files": files, "remaining": len(list_unlabeled())})


@app.route("/source/<path:name>")
def serve_source(name):
    return send_from_directory(SOURCE, name)


@app.route("/raw/<cls>/<path:name>")
def serve_raw(cls, name):
    return send_from_directory(RAW / cls, name)


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json()
    fname = data["filename"]
    cls = data["class"]
    if cls not in CLASSES:
        return jsonify({"error": f"unknown class {cls}"}), 400

    src = SOURCE / fname
    if not src.exists():
        return jsonify({"error": "file not found in source (already labeled?)"}), 404

    dst_dir = RAW / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_dir / fname))
    UNDO.append((fname, cls))
    return jsonify({"ok": True, "counts": class_counts(), "remaining": len(list_unlabeled())})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    if not UNDO:
        return jsonify({"error": "nothing to undo"}), 400
    fname, cls = UNDO.pop()
    src = RAW / cls / fname
    if not src.exists():
        return jsonify({"error": "file vanished"}), 404
    shutil.move(str(src), str(SOURCE / fname))
    return jsonify({
        "ok": True,
        "restored": {"filename": fname, "class": cls},
        "counts": class_counts(),
        "remaining": len(list_unlabeled()),
    })


@app.route("/api/stats")
def api_stats():
    return jsonify({"counts": class_counts(), "remaining": len(list_unlabeled())})


@app.route("/api/finalize", methods=["POST"])
def api_finalize():
    """
    Cap each class at max_per_class (random subsample of overflow goes to
    labels_discarded/<class>). Then split the kept crops into labels/train
    and labels/val by val_ratio.

    Wipes labels/ and labels_discarded/ first so this is idempotent.
    """
    data = request.get_json()
    cap = int(data["max_per_class"])
    val_ratio = float(data.get("val_ratio", 0.2))
    seed = int(data.get("seed", 42))
    rng = random.Random(seed)

    if OUT.exists():
        shutil.rmtree(OUT)
    if DISCARDED.exists():
        shutil.rmtree(DISCARDED)

    summary = {}
    for cls in CLASSES:
        src_dir = RAW / cls
        files = sorted(src_dir.glob("*"))
        rng.shuffle(files)

        keep = files[:cap]
        discard = files[cap:]

        # Parked overflow
        if discard:
            d = DISCARDED / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in discard:
                shutil.copy2(f, d / f.name)

        # Train/val split
        n_val = max(1, int(round(len(keep) * val_ratio))) if len(keep) >= 2 else 0
        val_files = keep[:n_val]
        train_files = keep[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            if not split_files:
                continue
            split_dir = OUT / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, split_dir / f.name)

        summary[cls] = {
            "total_labeled": len(files),
            "kept": len(keep),
            "discarded": len(discard),
            "train": len(train_files),
            "val": len(val_files),
        }

    return jsonify({"ok": True, "summary": summary,
                    "labels_dir": str(OUT), "discarded_dir": str(DISCARDED)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Flat folder of crops to label (e.g. crops/RGoldies_23_10_25_flat)")
    parser.add_argument("--raw-dir", default="labels_raw",
                        help="Where labeled crops live during labeling")
    parser.add_argument("--out-dir", default="labels",
                        help="Final train/val output (consumed by train_classifier.py)")
    parser.add_argument("--discarded-dir", default="labels_discarded")
    parser.add_argument("--classes", nargs="+", required=True,
                        help="Class names, e.g. --classes regular head_down ... unclear")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    global SOURCE, RAW, OUT, DISCARDED, CLASSES
    SOURCE = Path(args.source).resolve()
    RAW = Path(args.raw_dir).resolve()
    OUT = Path(args.out_dir).resolve()
    DISCARDED = Path(args.discarded_dir).resolve()
    CLASSES = list(args.classes)

    if not SOURCE.is_dir():
        raise SystemExit(f"Source dir not found: {SOURCE}")

    RAW.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (RAW / c).mkdir(parents=True, exist_ok=True)

    print(f"Source:     {SOURCE}")
    print(f"Raw labels: {RAW}")
    print(f"Output:     {OUT}")
    print(f"Discarded:  {DISCARDED}")
    print(f"Classes:    {CLASSES}")
    print(f"\nOpen http://{args.host}:{args.port}/ to start labeling.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

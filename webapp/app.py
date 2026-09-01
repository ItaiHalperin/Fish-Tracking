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
import re
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Populated in main()
SOURCE: Path = Path()
RAW: Path = Path()
OUT: Path = Path()
DISCARDED: Path = Path()
CROPS_ROOT: Path = Path()             # root of unflattened crops (for context strip)
CLASSES: list[str] = []
# Maps the normalized video name embedded in flat filenames back to the
# original folder under CROPS_ROOT (which may contain spaces).
VIDEO_DIR_MAP: dict[str, Path] = {}
# In-memory undo stack: (filename, class_it_was_moved_to)
UNDO: list[tuple[str, str]] = []

# Flat filename schema: "<video_norm>__id_<N>_frame_<M>.jpg"
FLAT_PATTERN = re.compile(r"^(?P<video>.+)__id_(?P<fid>\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$")

# Rare-class "neighborhood" mode parameters, set in main()
RARE_THRESHOLD = 30
NEIGHBORHOOD_SIZE = 10
# One-time priority list: filenames loaded from --seed-queue. Served before
# random sampling. Drains as the user labels (and as items leave SOURCE).
SEED_QUEUE: list[str] = []


def parse_flat(name: str) -> dict | None:
    m = FLAT_PATTERN.match(name)
    if not m:
        return None
    return {"video": m["video"], "fid": int(m["fid"]), "frame": int(m["frame"])}


def neighborhood_for(reference: str, n: int) -> list[str]:
    """
    Unlabeled crops in SOURCE that share (video, fish_id) with `reference`,
    sorted by frame-number proximity. Excludes the reference itself.
    """
    parsed = parse_flat(reference)
    if not parsed:
        return []
    pattern = f"{parsed['video']}__id_{parsed['fid']}_frame_*.jpg"
    hits = []
    for p in SOURCE.glob(pattern):
        if p.name == reference:
            continue
        pp = parse_flat(p.name)
        if not pp:
            continue
        hits.append((abs(pp["frame"] - parsed["frame"]), p.name))
    hits.sort()
    return [name for _, name in hits[:n]]


def build_video_dir_map() -> dict[str, Path]:
    """
    crops/all_flat encodes video names with spaces replaced by underscores.
    The originals (e.g. 'NRGoldies 3_11_25') may still have spaces. Map the
    underscore version back to the real directory.
    """
    out = {}
    if not CROPS_ROOT.is_dir():
        return out
    for d in CROPS_ROOT.iterdir():
        if not d.is_dir() or d.name.endswith("_flat") or d.name == "all_flat":
            continue
        out[d.name.replace(" ", "_")] = d
    return out


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
    """
    Return up to N unlabeled filenames. If SEED_QUEUE has entries that are
    still in SOURCE, drain them first (they're the priority items from
    --seed-queue). Then fill the rest with random samples.
    """
    n = int(request.args.get("n", 50))
    available = set(list_unlabeled())

    picks: list[str] = []
    # Drain seed queue (in order, skipping any that have been labeled away).
    while SEED_QUEUE and len(picks) < n:
        f = SEED_QUEUE.pop(0)
        if f in available:
            picks.append(f)
            available.discard(f)

    n_more = n - len(picks)
    pool = list(available)
    if n_more > 0:
        if len(pool) <= n_more:
            picks.extend(pool)
        else:
            picks.extend(random.sample(pool, n_more))

    return jsonify({"files": picks, "remaining": len(list_unlabeled())})


@app.route("/api/context")
def api_context():
    """
    Return up to ±N frames around the requested crop for the same fish ID.
    Each entry is {url, frame, is_center}. The frontend renders them as a
    context strip; only the center one gets labeled.
    """
    fname = request.args.get("filename", "")
    radius = int(request.args.get("radius", 2))
    parsed = parse_flat(fname)
    if not parsed:
        return jsonify({"frames": []})
    real_dir = VIDEO_DIR_MAP.get(parsed["video"])
    if not real_dir:
        return jsonify({"frames": []})
    fish_dir = real_dir / f"id_{parsed['fid']}"
    if not fish_dir.is_dir():
        return jsonify({"frames": []})

    center = parsed["frame"]
    out = []
    for offset in range(-radius, radius + 1):
        f = offset + center
        crop = fish_dir / f"frame_{f}.jpg"
        if not crop.exists():
            continue
        out.append({
            "url": f"/context-img/{parsed['video']}/{parsed['fid']}/{f}",
            "frame": f,
            "is_center": offset == 0,
        })
    return jsonify({"frames": out, "video": parsed["video"], "fid": parsed["fid"]})


@app.route("/context-img/<video>/<int:fid>/<int:frame>")
def serve_context_image(video, fid, frame):
    real_dir = VIDEO_DIR_MAP.get(video)
    if not real_dir:
        return ("not found", 404)
    return send_from_directory(real_dir / f"id_{fid}", f"frame_{frame}.jpg")


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

    counts = class_counts()
    # If this class is rare, fetch nearby crops of the same fish so the user
    # can label that pose in bulk.
    neighborhood: list[str] = []
    if cls != "unclear" and counts.get(cls, 0) < RARE_THRESHOLD:
        neighborhood = neighborhood_for(fname, NEIGHBORHOOD_SIZE)

    return jsonify({
        "ok": True,
        "counts": counts,
        "remaining": len(list_unlabeled()),
        "neighborhood": neighborhood,
        "neighborhood_reason": cls if neighborhood else None,
    })


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
    test_ratio = float(data.get("test_ratio", 0.0))
    seed = int(data.get("seed", 42))
    if val_ratio + test_ratio >= 1.0:
        return jsonify({"error": "val_ratio + test_ratio must be < 1"}), 400
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

        if discard:
            d = DISCARDED / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in discard:
                shutil.copy2(f, d / f.name)

        n = len(keep)
        n_val = int(round(n * val_ratio)) if n >= 2 else 0
        n_test = int(round(n * test_ratio)) if n >= 2 else 0
        val_files = keep[:n_val]
        test_files = keep[n_val:n_val + n_test]
        train_files = keep[n_val + n_test:]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if not split_files:
                continue
            split_dir = OUT / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, split_dir / f.name)

        summary[cls] = {
            "total_labeled": len(files),
            "kept": n,
            "discarded": len(discard),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
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
    parser.add_argument("--crops-root", default="crops",
                        help="Root with unflattened crops (used for the context strip). "
                             "Should contain <video>/id_<N>/frame_<M>.jpg trees.")
    parser.add_argument("--rare-threshold", type=int, default=30,
                        help="Below this count, labeling a class triggers neighborhood mode")
    parser.add_argument("--neighborhood-size", type=int, default=10,
                        help="How many neighboring same-fish crops to surface after a rare label")
    parser.add_argument("--seed-queue", default=None,
                        help="Optional text file with one filename per line. Served before random sampling.")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names. If omitted, derived from webapp/static/reference/<class>.{png,jpg} "
                             "plus a final 'unclear' bin.")
    parser.add_argument("--reject-class", default="unclear",
                        help="Name of the always-appended reject class (no reference image expected).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    global SOURCE, RAW, OUT, DISCARDED, CROPS_ROOT, CLASSES, VIDEO_DIR_MAP, RARE_THRESHOLD, NEIGHBORHOOD_SIZE, SEED_QUEUE
    RARE_THRESHOLD = args.rare_threshold
    NEIGHBORHOOD_SIZE = args.neighborhood_size
    if args.seed_queue:
        seed_path = Path(args.seed_queue)
        if seed_path.is_file():
            SEED_QUEUE = [line.strip() for line in seed_path.read_text().splitlines() if line.strip()]
            print(f"Seed queue: loaded {len(SEED_QUEUE)} filenames from {seed_path}")
        else:
            print(f"WARNING: --seed-queue file not found: {seed_path}")
    SOURCE = Path(args.source).resolve()
    RAW = Path(args.raw_dir).resolve()
    OUT = Path(args.out_dir).resolve()
    DISCARDED = Path(args.discarded_dir).resolve()
    CROPS_ROOT = Path(args.crops_root).resolve()
    VIDEO_DIR_MAP = build_video_dir_map()

    if args.classes:
        CLASSES = list(args.classes)
    else:
        ref_dir = Path(__file__).parent / "static" / "reference"
        derived = sorted({p.stem for p in ref_dir.glob("*")
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"}})
        if not derived:
            raise SystemExit(f"No --classes given and no reference images in {ref_dir}")
        if args.reject_class not in derived:
            derived.append(args.reject_class)
        CLASSES = derived
        print(f"Derived {len(CLASSES)} classes from {ref_dir}")

    if not SOURCE.is_dir():
        raise SystemExit(f"Source dir not found: {SOURCE}")

    RAW.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (RAW / c).mkdir(parents=True, exist_ok=True)

    print(f"Source:     {SOURCE}")
    print(f"Raw labels: {RAW}")
    print(f"Output:     {OUT}")
    print(f"Discarded:  {DISCARDED}")
    print(f"Crops root: {CROPS_ROOT}  ({len(VIDEO_DIR_MAP)} videos indexed for context)")
    print(f"Classes:    {CLASSES}")
    print(f"\nOpen http://{args.host}:{args.port}/ to start labeling.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

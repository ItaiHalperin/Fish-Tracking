"""
Heading-line annotator.

Upgrades the heading of already-labeled crops from coarse class-centers to a
continuous angle. Roll is NOT touched — it stays derived from each crop's class
folder via configs/angle_map.yaml, so you only annotate the one axis that needs
precision.

You see a crop with an arrow pre-filled at its class-center heading; click (or
drag) toward where the fish's head points to correct it, Enter to save + advance.
Output: a single {crop_basename: heading_deg} JSON the angle_reg model reads.

Usage:
    python webapp/angle_app.py --raw-dir labels_raw --out labels_angles/headings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

RAW = Path("labels_raw")
OUT = Path("labels_angles/headings.json")
ANGLE_MAP: dict[str, float] = {}          # class -> class-center heading
FILE_CLASS: dict[str, str] = {}           # crop basename -> class
WORKLIST: list[str] = []                  # ordered crop basenames
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _load_headings() -> dict[str, float]:
    if OUT.is_file():
        return {k: float(v) for k, v in json.loads(OUT.read_text()).items()}
    return {}


def _save_heading(name: str, heading: float) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = _load_headings()
    data[name] = round(float(heading) % 360.0, 1)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=0, sort_keys=True))
    tmp.replace(OUT)


@app.route("/")
def index():
    return render_template("angle.html")


@app.route("/api/worklist")
def worklist():
    saved = _load_headings()
    items = [{
        "file": name,
        "cls": FILE_CLASS[name],
        "prefill": saved.get(name, ANGLE_MAP.get(FILE_CLASS[name], 0.0)),
        "done": name in saved,
    } for name in WORKLIST]
    return jsonify({"items": items, "n_done": len(saved), "total": len(WORKLIST)})


@app.route("/img/<path:name>")
def img(name: str):
    cls = FILE_CLASS.get(name)
    if not cls:
        return "not found", 404
    return send_file((RAW / cls / name).resolve())


@app.route("/api/save", methods=["POST"])
def save():
    data = request.get_json()
    _save_heading(data["file"], data["heading"])
    return jsonify({"ok": True})


def main():
    global RAW, OUT, ANGLE_MAP, FILE_CLASS, WORKLIST
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="labels_raw")
    ap.add_argument("--angle-map", default="configs/angle_map.yaml")
    ap.add_argument("--out", default="labels_angles/headings.json")
    ap.add_argument("--skip-class", action="append", default=["unclear"])
    ap.add_argument("--port", type=int, default=5002)
    args = ap.parse_args()

    RAW = Path(args.raw_dir)
    OUT = Path(args.out)
    raw_map = yaml.safe_load(Path(args.angle_map).read_text())
    ANGLE_MAP = {cls: float(v["heading"]) for cls, v in raw_map.items()}

    skip = set(args.skip_class)
    for cls_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        if cls_dir.name in skip:
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in _IMG_EXT:
                FILE_CLASS[f.name] = cls_dir.name
                WORKLIST.append(f.name)

    print(f"Heading annotator: {len(WORKLIST)} crops from {RAW}")
    print(f"  writing headings to {OUT}")
    print(f"  open http://127.0.0.1:{args.port}")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()

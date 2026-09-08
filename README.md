# Fish Orientation — Running the Tools

End-to-end system for measuring fish orientation from tank video: detect & track
fish → crop each one per frame → classify orientation (roll, or the full
pose × facing position). This README covers **how to run** the roll pipeline — the project's
deliverable — end to end. The models it serves are described in
[ml_storage/README.md](ml_storage/README.md); superseded models are in
[archive/README.md](archive/README.md).

A sample video of the aquarium can be found in the following link: https://drive.google.com/file/d/1McPh04mV62h5c5SnCH8ucwG3Kr8fTaHk/view?usp=sharing

## Setup

- **Use a virtual environment.** A plain `pip install` against a Homebrew or
  system Python fails with `error: externally-managed-environment` (PEP 668) and
  installs nothing:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  pip install -r requirements.txt    # pinned; see requirements.txt
  ```

  Keep it activated, or call `.venv/bin/python` explicitly — running a bare
  `python3` picks up the system interpreter, where the deps are absent.
- **ffmpeg** on PATH — a system dependency, not pip:

  ```bash
  brew install ffmpeg           # macOS
  sudo apt install ffmpeg       # Debian / Ubuntu
  winget install Gyan.FFmpeg    # Windows
  ```

  Verify with `ffmpeg -version`. It is used to read non-mp4 input (`.MTS`,
  `.wmv`, …) **and** whenever `--start` or `--duration` trims a clip — including
  from an `.mp4`, so the examples below need it. Without it you get
  `FileNotFoundError: 'ffmpeg'`.
- **Device:** examples use `mps` (Apple GPU). Use `--device cpu` if you have no
  GPU, or `cuda` on NVIDIA. Anything unavailable falls back to CPU automatically.
- Run commands from the repo root. For the `classifier.*` module commands, prefix
  with `PYTHONPATH=.` if the package isn't installed.
- Long jobs: prefix with `caffeinate -i` on macOS so the machine doesn't sleep
  mid-run.

---

## 1. Analysis webapp — `webapp/analyze_app.py`

Browser front-end for the full pipeline: drop in a video (or paste a path), it
runs track → crop → classify in the background and shows the per-fish + pooled
breakdown live. One job at a time (single GPU).

| Flag | Default | Purpose |
|---|---|---|
| `--device` | mps | `mps` / `cpu` / `cuda` |
| `--port` / `--host` | 5070 / 127.0.0.1 | Server address |

```bash
python webapp/analyze_app.py --device mps
# open http://127.0.0.1:5070
```

---

## 2. Full pipeline — `analysis/pipeline.py`

Runs the whole thing on one video: **track → crop → classify → summarise**.
Writes `summary.xlsx` (per-fish + AVERAGE roll/position %) and `traces.xlsx`
(per-frame labels). Auto-remuxes non-mp4 formats and can trim to a clip.

| Flag | Default | Purpose |
|---|---|---|
| `video` (positional) | — | Input video path |
| `-o, --output-dir` | required | Where `summary.xlsx` / `traces.xlsx` go |
| `--duration` | full video | Clip length in **seconds** (e.g. 300 = first 5 min) |
| `--start` | 0 | Start offset in seconds |
| `--device` | mps | `mps` / `cpu` / `cuda` |
| `--crops-dir` | `crops/<stem>` | Where per-fish crops are written |
| `--track-output-dir` | none | Also save an annotated tracking video (boxes + IDs) |
| `--conf` | 0.4 | Detection confidence threshold |
| `--smooth-window` | 5 | Temporal majority-vote smoothing of labels |
| `--storage` | `ml_storage` | Model registry (resolves detector + classifier) |

The classifier is whatever is registered as `fish_position_classifier` in
`ml_storage` (currently the **roll** model). Example — first 5 minutes on MPS:

```bash
caffeinate -i python analysis/pipeline.py "videos/RGoldies 18_9_25.mp4" \
  -o results_pipeline/RGoldies_18_9_25 --duration 300 --device mps
```

---

## 3. Making crops to label — `dataset_tools/`

The labeler and the heading annotator (below) consume **crops**, so a fresh
checkout has nothing to feed them. Sections 1 and 2 produce crops as a
side-effect (`--crops-dir`, default `crops/<video-stem>/`), which is the easiest
source. To make them directly:

```bash
# 1. Detect + track a video and write one folder of crops per fish ID.
#    Uses the registered detector unless you pass --weights.
python -m dataset_tools.extract_crops --video "videos/RGoldies 18_9_25.mp4" \
    --duration 300 --output_dir crops
# -> crops/RGoldies 18_9_25/id_<N>/frame_<M>.jpg

# 2. Flatten into the single folder the labeler expects.
python -m dataset_tools.flatten_crops --crops-dir "crops/RGoldies 18_9_25"
# -> crops/RGoldies 18_9_25_flat/
```

| Command | Purpose |
|---|---|
| `dataset_tools.extract_crops` | Video → `crops/<video>/id_N/frame_M.jpg` (10% padding, matching training) |
| `dataset_tools.flatten_crops` | `crops/<video>/` → flat `crops/<video>_flat/` for `--source` |
| `dataset_tools.sample_frames` | Sample whole frames from videos, for **box** annotation |
| `dataset_tools.ingest_coco` / `ingest_yolo` | Import annotated frames (Roboflow COCO / YOLO exports) into `dataset/` |
| `detection.train` | Train the detector on `dataset/` (`--model s --version 26`) |
| `detection.track` | Track only, saving an annotated video (`--use-storage`) |

Both trees are worth keeping: the labeler resolves a crop's neighbouring frames
from the unflattened `crops/<video>/id_N/` tree to show its context strip. See
[webapp/README.md](webapp/README.md) for the labeling workflow end to end.

---

## 4. Labeler webapp — `webapp/app.py`

Hand-label crop orientation classes. Shows a crop (with a context strip of
neighboring frames); each label **moves** the crop into `labels_raw/<class>/`.
The `/finish` page caps per-class counts and splits into `labels/train|val`.

> ⚠️ The classifier pipeline reads `labels_split/` (built by `classifier.data`),
> **not** `labels/`. The labeler owns `labels/`; don't point training at it.

| Flag | Default | Purpose |
|---|---|---|
| `--source` | required | Flat folder of crops to label — the `_flat` folder from §3 |
| `--raw-dir` | `labels_raw` | Where labeled crops are moved, per class |
| `--seed-queue` | none | Text file of filenames to serve first (from `classifier.mine_candidates`) |
| `--rare-threshold` | 30 | Classes below this count are treated as rare |
| `--neighborhood-size` | 10 | Context-strip neighbor count |
| `--reject-class` | unclear | Bucket for "can't tell" |
| `--port` / `--host` | 5050 / 127.0.0.1 | Server address |

Continuing the example from §3:

```bash
python webapp/app.py --source "crops/RGoldies 18_9_25_flat"
# open http://127.0.0.1:5050
```

---

## Classifier CLIs (training / evaluation)

Secondary, but the pipeline depends on models these produce. All take
`PYTHONPATH=.`.

| Command | Purpose | Example |
|---|---|---|
| `classifier.data` | Build leak-free, class-aware split | `python -m classifier.data --out labels_split --skip-class unclear` |
| `classifier.train` | Train a model from a YAML config | `python -m classifier.train --config configs/roll_cls.yaml` |
| `classifier.test_roll` | Test a roll run on the held-out split | `python -m classifier.test_roll --run results/roll_cls_<ts>` |
| `classifier.baselines` | Majority + logistic-regression floors | `python -m classifier.baselines --data-dir labels_split` |
| `classifier.infer` | Classify a pre-extracted crops folder → Excel | `python -m classifier.infer --run archive/classifier_experiments/<run> --crops-dir crops/<video>` |

Configs live in `configs/` (one YAML per run). For roll: `roll_cls_angular.yaml`
is the deliverable, `roll_cls.yaml` its plain-loss ablation. Training writes a
fresh run folder under `results/`; completed runs are kept in
[`archive/`](archive/README.md).

The full-position (16-way) and angle-regression models, their configs, the
active-learning miner and the matching test harnesses are covered in
[classifier/README.md](classifier/README.md) — none of them are needed for roll.

`classifier.test_roll` also reports **angular error** — how many degrees off the
roll was, not just whether the class matched — into `roll_test_report.json`.
Re-running it on an old run re-scores it with no retraining.

Composite (16-way) and angle-regression runs use a different harness,
`classifier.test`; see [classifier/README.md](classifier/README.md) for which
harness matches which `model_type`.

Any config can charge mistakes by their angular size instead of a flat
right/wrong by setting `angular_tau` (degrees; `0` = off, reproducing the plain
loss exactly).

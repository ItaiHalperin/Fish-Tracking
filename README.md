# Fish Orientation — Running the Tools

End-to-end system for measuring fish orientation from tank video: detect & track
fish → crop each one per frame → classify orientation (roll, or the full
pose × facing position). This README covers **how to run** the pipeline and the
three webapps. The models it serves are described in
[ml_storage/README.md](ml_storage/README.md); superseded models are in
[archive/README.md](archive/README.md).

A sample video of the aquarium can be found in the following link: https://drive.google.com/file/d/1McPh04mV62h5c5SnCH8ucwG3Kr8fTaHk/view?usp=sharing

## Setup

- Install Python deps: `pip install -r requirements.txt` (pinned to the working
  environment; see [requirements.txt](requirements.txt)).
- **ffmpeg** on PATH — a system dependency, not pip (only needed to read/trim
  `.MTS` and other non-mp4 formats).
- **Device:** examples use `mps` (Apple GPU). Use `--device cpu` if you have no
  GPU, or `cuda` on NVIDIA. Anything unavailable falls back to CPU automatically.
- Run commands from the repo root. For the `classifier.*` module commands, prefix
  with `PYTHONPATH=.` if the package isn't installed.
- Long jobs: prefix with `caffeinate -i` on macOS so the machine doesn't sleep
  mid-run.

---

## 1. Full pipeline — `analysis/pipeline.py`

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

## 2. Labeler webapp — `webapp/app.py`

Hand-label crop orientation classes. Shows a crop (with a context strip of
neighboring frames); each label **moves** the crop into `labels_raw/<class>/`.
The `/finish` page caps per-class counts and splits into `labels/train|val`.

> ⚠️ The classifier pipeline reads `labels_split/` (built by `classifier.data`),
> **not** `labels/`. The labeler owns `labels/`; don't point training at it.

| Flag | Default | Purpose |
|---|---|---|
| `--source` | required | Flat folder of crops to label (e.g. `crops/all_flat`) |
| `--raw-dir` | `labels_raw` | Where labeled crops are moved, per class |
| `--seed-queue` | none | Text file of filenames to serve first (e.g. from the miner) |
| `--rare-threshold` | 30 | Classes below this count are treated as rare |
| `--neighborhood-size` | 10 | Context-strip neighbor count |
| `--reject-class` | unclear | Bucket for "can't tell" |
| `--port` / `--host` | 5050 / 127.0.0.1 | Server address |

```bash
python webapp/app.py --source crops/all_flat --seed-queue seed_queue.txt
# open http://127.0.0.1:5050
```

---

## 3. Heading annotator — `webapp/angle_app.py`

Draw the head-direction line on already-labeled crops to upgrade heading from
class-center to a **continuous angle** (roll is left untouched). Writes
`{crop: heading_deg}` JSON consumed by the `angle_reg` model.

| Flag | Default | Purpose |
|---|---|---|
| `--raw-dir` | `labels_raw` | Crops to annotate (read from class folders) |
| `--angle-map` | `configs/angle_map.yaml` | Class → (heading, roll); seeds the pre-filled arrow |
| `--out` | `labels_angles/headings.json` | Output annotations |
| `--skip-class` | `unclear` | Class(es) to skip (repeatable) |
| `--port` | 5002 | Server port |

```bash
python webapp/angle_app.py --raw-dir labels_raw --out labels_angles/headings.json
# open http://127.0.0.1:5002  (click head direction, Enter to save+next)
```

---

## 4. Analysis webapp — `webapp/analyze_app.py`

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

## Classifier CLIs (training / evaluation)

Secondary, but the pipeline depends on models these produce. All take
`PYTHONPATH=.`.

| Command | Purpose | Example |
|---|---|---|
| `classifier.data` | Build leak-free, class-aware split | `python -m classifier.data --out labels_split --skip-class unclear` |
| `classifier.train` | Train a model from a YAML config | `python -m classifier.train --config configs/roll_cls.yaml` |
| `classifier.test` | Test composite classification | `python -m classifier.test --run archive/classifier_experiments/<run>` |
| `classifier.test_roll` | Test roll (4-way) | `python -m classifier.test_roll --run archive/classifier_experiments/<run>` |
| `classifier.baselines` | Majority + logistic-regression floors | `python -m classifier.baselines --data-dir labels_split` |
| `classifier.mine_candidates` | Find rare-class crops to label (active learning) | `python -m classifier.mine_candidates --run archive/classifier_experiments/<run> --top 200` |
| `classifier.infer` | Classify a pre-extracted crops folder → Excel | `python -m classifier.infer --run archive/classifier_experiments/<run> --crops-dir crops/<video>` |

Configs live in `configs/` (one YAML per run): `roll_cls_angular.yaml` (roll, the
goal — the deliverable), `multihead_decoupled.yaml` (best full-position
classifier), `angle_reg.yaml` (continuous angles), plus their plain-loss ablation
partners and baselines. Training writes a fresh run folder under `results/`;
completed runs are kept in [`archive/`](archive/README.md).

`classifier.test` and `classifier.test_roll` also report **angular error** — how
many degrees off the orientation was, not just whether the class matched — into
`test_report.json` / `roll_test_report.json`. Heading truth comes from
`labels_angles/headings.json`; pass `--headings` to point elsewhere. Re-running
either command on an old run re-scores it with no retraining.

Any config can charge mistakes by their angular size instead of a flat
right/wrong by setting `angular_tau` (degrees; `0` = off, reproducing the plain
loss exactly).

# Classification improvement plan

Working doc. Phases ordered by dependency, not by "what's coolest". Each phase
has a clear owner (✦ me, ▲ you, ◇ both) and a definition of done.

---

## Phase 0 — Reorganize the code  ✦

**Why first**: every other phase touches training/inference. If the layout is
messy now, it's worse after we add baselines, sweeps, and configs.

**What changes**

```
classifier/
  __init__.py
  config.py           # dataclasses for TrainConfig, AugConfig, etc.
  data.py             # label dir scanning, balanced sampling, splits
  models/
    __init__.py
    base.py           # ABC: Classifier with .train/.predict/.save/.load
    yolo_cls.py       # current yolo11n-cls implementation
    sklearn_baseline.py   # Phase 4
  train.py            # entrypoint: reads config, dispatches to model
  test.py             # benchmarking (moves test_classifier.py here)
  infer.py            # video → Excel (moves classify_video.py here)
configs/
  yolo_n_default.yaml
  yolo_n_heavy_aug.yaml
  sklearn_resnet_features.yaml
results/
  <config_name>_<timestamp>/
    weights/...
    config.yaml          # copy of the config used
    test_report.json     # accuracy, F1, confusion matrix
    notes.md             # human notes after the run
```

Old root-level scripts (`train_classifier.py`, `test_classifier.py`,
`classify_video.py`, `extract_crops.py`, `flatten_crops.py`) stay where they
are as thin shims that delegate, so existing commands keep working.

**Definition of done**: `python -m classifier.train --config configs/yolo_n_default.yaml`
produces a `results/yolo_n_default_<ts>/` folder with weights + test report,
and the existing commands still work.

---

## Phase 1 — Smarter labeling: random + context  ✦

**Why**: 100 sequential crops of the same fish in the same pose are 100x less
useful than 100 from random tracks/frames. Adjacency makes the *single*
picked frame easier to label because you can see what's happening before/after.

**What changes**

- `webapp/app.py`: `/api/queue` returns a random sample from the source
  folder instead of `sorted()[:n]`. The label page already calls it; just
  changes selection.
- `label.html` gets a **context strip** above the main image: 5 thumbnails
  showing frames at the same fish ID surrounding the picked one (e.g. ±2).
  The center thumbnail is highlighted; only it gets labeled.
- Crop filenames already encode `video__id_N_frame_M.jpg`, so finding the
  neighbors is just a `glob` and a sort.

**Caveat**: the context only works for the original `crops/<video>/id_N/`
trees, not the flattened pool. The app will resolve neighbors by parsing
the filename and reading from the unflattened tree. Both trees need to stick
around (small cost — they're crops not videos).

**Definition of done**: launching the labeler shows the picked crop with up
to 4 neighbors above it; the keyboard/click only labels the picked one.

---

## Phase 2 — Augmentation, but carefully  ✦

**Why**: ultralytics already augments. The current train script correctly
disables horizontal flip (it would swap `_facing_left` ↔ `_facing_right`).
But there's more nuance.

**Safe knobs to expose in the config**:
- `degrees` — small rotation (≤15°) is fine; large rotations change labels
  (a 90° rotation of `head_up_facing_left` becomes `regular_facing_…`).
- `translate`, `scale` — safe, the fish is centered.
- `hsv_h`, `hsv_s`, `hsv_v` — safe.
- `mosaic` — safe (just composites multiple crops).
- `erasing` — random pixel-block erase, good for robustness.

**Hard-coded off**:
- `fliplr` (already off — direction-changing).
- `flipud` (off — pitch-changing).
- Large `degrees` — would change class.

Config option `direction_aware: true` keeps the strict caps; setting it to
`false` (only meaningful if you switch to direction-agnostic classes later)
opens up flips/rotations.

**Definition of done**: `configs/yolo_n_heavy_aug.yaml` exists with documented
augmentation values; results from running it vs default land in `results/`.

---

## Phase 3 — Config-driven sweep + results doc  ◇

**Why**: "try different parameters and see which works" needs a system or
you lose track in 3 runs.

**What changes** (mostly already from Phases 0-2):
- Each config is one YAML file. Fields: `model`, `epochs`, `imgsz`, `batch`,
  `lr0`, augmentation block, `val_ratio`, `seed`, etc.
- `python -m classifier.train --config <path>` is the single entrypoint.
- After training, `python -m classifier.test --results results/<name>` runs
  the test set and writes `test_report.json` next to the weights.
- A `results/INDEX.md` table (auto-appended after each run) summarizes:
  config name, train size, test acc, macro F1, weak classes, timestamp.

**Your part ▲**:
- Define 4-6 configs to compare (I'll draft the first set; you tweak).
- Run them, look at `INDEX.md`, write 1-line notes in each run's `notes.md`.
- Decide which to keep.

**Definition of done**: at least 3 runs in `results/`, `INDEX.md` populated,
you can point to a "current best" config.

---

## Phase 4 — Baseline comparison ✦ (with ▲ to validate)

**Why**: you can't claim your model is good without something to beat.
"95% accuracy" is meaningless without knowing what a dumb baseline gets.

**Concrete baselines, in order of cost**:
1. **Majority class** — always predict the most common class. 1 line of
   code. Establishes the floor.
2. **Logistic regression on raw pixels** — flatten + scale + sklearn
   LogisticRegression. Cheap. Likely terrible. Bounds "is the task
   non-trivial".
3. **Logistic regression on YOLO embeddings** — use the YOLO detector's
   feature backbone (`best.pt` from your partner) to get a 512-d embedding
   per crop, then sklearn on top. This is the strong baseline. If your
   yolo-cls model can't beat this, you have a problem.

These slot into the `models/` ABC from Phase 0. Each gets a config.

**Definition of done**: `results/INDEX.md` has rows for at least baselines
#1 and #3 alongside your yolo-cls runs.

---

## Phase 5 — Angle regression  ◇  (decision point, not a free addition)

**Read this before saying yes.** It is *not* "another class scheme" —
it's a different ML problem.

**What it would mean**:
- Labels become continuous: e.g. `pitch ∈ [0, 360)` (body axis direction)
  and `roll ∈ [0, 360)` (rotation around body axis, where 180° = upside
  down). Possibly also a binary `from_above` flag for top-view crops.
- Labeling UI: replaced with two angle dials per crop. Each label takes
  *seconds longer* than a button click. Throughput drops 3-5x.
- Model output head: 2 regression heads with cosine/sine encoding
  (to handle the wrap-around at 360°).
- Loss: angular distance, not cross-entropy.
- Reporting: "% time upside down" becomes "% time with |roll - 180°| <
  threshold" — much cleaner than summing classes.
- Existing labels: not reusable directly. You'd need to map each class to a
  canonical (pitch, roll) and possibly relabel a sample manually.

**When this is right**: when you've squeezed everything out of the
classifier and the per-class taxonomy is fighting you (e.g. fish that are
mid-rotation get force-binned into nearest class incorrectly).

**When this is wrong**: right now, before you've maxed out the simpler
approach. You have only ~250 labeled crops; the classifier can still
improve a lot.

**My recommendation**: park this. Revisit after Phase 4 if class accuracy
plateaus and you observe systematic problems with the discretization.

---

## What I'll do vs what you do

| Step | Me ✦ | You ▲ | Notes |
|---|---|---|---|
| 0. Reorganize | ✦ | review the new layout | ~1 hour of refactor |
| 1. Random + context labeler | ✦ | label a fresh batch with it | better diversity = better model |
| 2. Aug configs | ✦ |  | knobs are exposed; you tune later |
| 3. Sweep | ✦ scaffold | ▲ run configs, write notes | this is where your judgment matters most |
| 4. Baselines | ✦ | ▲ confirm numbers match expectations | |
| 5. Angles | — | ▲ decide go / no-go | only after seeing Phase 3-4 results |

---

## Decisions (locked in)

- Phase order: 0 → 1 → 2 → 3 → 4. Phase 5 parked.
- Old root-level scripts go away during reorganize. No back-compat shims.
- Phase 4 baseline = LR on raw pixels only. Skip majority + YOLO-embedding LR.
- Labels start fresh: existing `labels_raw/` archived to `labels_raw_v1/`,
  new labeling begins from empty after Phase 1 lands.
- Phase 2 augmentation: implement both
  - small continuous rotation (≤15°) without label change, and
  - cardinal-aligned rotation (45°/90°/...) with label remapping.

## Labeling rounds

### Round 1 — 2026-06-19 (snapshot: `labels_raw_snapshot_2026-06-19/`)

2,820 labels total. **1,950 (69%) are `unclear`** — heavy reject bin.
Of the ~870 useful labels, the distribution is very skewed.

| Class | Count | Notes |
|---|---:|---|
| unclear | 1,950 | dominates — `infer.py` excludes this from reported percentages |
| regular_facing_right | 169 | |
| diag_up_left_facing_down | 125 | |
| regular_facing_left | 119 | |
| head_up_facing_left | 107 | |
| diag_down_right_facing_down | 103 | |
| head_down_facing_right | 70 | |
| diag_up_right_facing_down | 54 | |
| head_down_facing_left | 44 | |
| upside_down_facing_right | 26 | |
| diag_down_left_facing_down | 21 | |
| head_up_facing_right | 9 | weak |
| diag_up_right_facing_up | 8 | weak |
| upside_down_facing_left | 8 | weak |
| diag_down_left_facing_up | 6 | weak |
| diag_up_left_facing_up | 1 | unusable — drop or rebalance |
| diag_down_right_facing_up | 0 | unusable — drop or skip with `--skip-class` |

**Observations for reasoning later:**
- 2 classes (`diag_down_right_facing_up`, `diag_up_left_facing_up`) effectively
  don't exist in our data. Either prune them from the reference set or label
  more crops of those poses next round.
- 4 more classes (`head_up_facing_right`, `diag_up_right_facing_up`,
  `upside_down_facing_left`, `diag_down_left_facing_up`) have <10 samples — too
  few to validate. Train-set noise.
- "Upside-down" classes are very rare (8 + 26 = 34 of 870 useful = 4%). The
  motivating metric for this project is "% time upside down" — that's
  consistent with the data, but it means our test of the upside-down
  classification will be statistically thin.
- "Facing right" / "facing left" pairs are imbalanced (regular: 119/169,
  diag_up: 125/54, head_up: 107/9). Either the tank has natural directionality
  bias, or labeling habits.

### Round 2 — 2026-06-19 (snapshot: `labels_raw_snapshot_2026-06-19_v2/`)

2,984 labels total (+164 since round 1). Class deltas:

| Class | R1 | R2 | Δ | Notes |
|---|---:|---:|---:|---|
| unclear | 1,950 | 2,037 | +87 | still dominant |
| regular_facing_right | 169 | 191 | +22 | |
| diag_up_left_facing_down | 125 | 143 | +18 | |
| regular_facing_left | 119 | 128 | +9 | |
| diag_down_right_facing_down | 103 | 113 | +10 | |
| head_up_facing_left | 107 | 111 | +4 | |
| head_down_facing_right | 70 | 74 | +4 | |
| diag_up_right_facing_down | 54 | 56 | +2 | |
| head_down_facing_left | 44 | 48 | +4 | |
| diag_up_right_facing_up | 8 | 10 | +2 | still weak |
| diag_down_left_facing_down | 21 | 22 | +1 | |
| diag_down_right_facing_up | 0 | 1 | +1 | still unusable |
| diag_up_left_facing_up | 1 | 1 | 0 | still unusable |
| diag_down_left_facing_up | 6 | 6 | 0 | still weak |
| head_up_facing_right | 9 | 9 | 0 | still weak |
| upside_down_facing_right | 26 | 26 | 0 | |
| upside_down_facing_left | 8 | 8 | 0 | still weak |

**Takeaway:** rare classes are still rare. We either need to actively seek out
those poses next round (e.g. eyeball videos for upside-down/head-up-right
fish and label only those tracks) or the videos genuinely don't have enough
of them and we should drop/merge those classes before training.

### Round 3 — 2026-06-19 (snapshot: `labels_raw_snapshot_2026-06-19_v3/`)

3,391 labels total (+407 since R2). First round with **neighborhood-mode**
enabled (after a label on any class with <30 samples, the next 10 same-fish
crops were pulled into the queue).

| Class | R2 | R3 | Δ | Notes |
|---|---:|---:|---:|---|
| unclear | 2,037 | 2,246 | +209 | |
| regular_facing_right | 191 | 217 | +26 | |
| diag_up_left_facing_down | 143 | 163 | +20 | |
| regular_facing_left | 128 | 158 | +30 | |
| diag_down_right_facing_down | 113 | 128 | +15 | |
| head_up_facing_left | 111 | 124 | +13 | |
| head_down_facing_right | 74 | 83 | +9 | |
| diag_up_right_facing_down | 56 | 62 | +6 | |
| head_down_facing_left | 48 | 54 | +6 | |
| upside_down_facing_right | 26 | 40 | **+14** | crossed 30 mark |
| diag_down_left_facing_down | 22 | 41 | **+19** | crossed 30 mark |
| diag_down_left_facing_up | 6 | 29 | **+23** | almost crossed; was weakest after the *_up diagonals |
| head_up_facing_right | 9 | 18 | **+9** | doubled |
| upside_down_facing_left | 8 | 16 | **+8** | doubled |
| diag_up_right_facing_up | 10 | 10 | 0 | unchanged — neighborhoods didn't surface any new ones |
| diag_down_right_facing_up | 1 | 1 | 0 | still effectively absent |
| diag_up_left_facing_up | 1 | 1 | 0 | still effectively absent |

**Did neighborhood mode help?** Yes, substantially, but only for poses that
*exist in the videos*:

- **Classes <30 samples: 8 in R2 → 6 in R3.** Two classes
  (`diag_down_left_facing_down`, `upside_down_facing_right`) crossed the
  threshold and won't trigger neighborhood mode anymore.
- **`diag_down_left_facing_up`: 6 → 29** is the headline win — almost +400%
  on a class that was barely real before.
- **Both upside_down classes grew** (8→16, 26→40), and the right-facing one
  is now usable.
- **3 classes saw no help at all** (`diag_up_right_facing_up`,
  `diag_down_right_facing_up`, `diag_up_left_facing_up`). When a class is
  truly absent in the videos, neighborhood mode has nothing to surface — the
  fish doesn't enter that pose, so there are no nearby same-fish frames to
  label. These three should probably be dropped before training (or merged
  with their `_down` counterparts since the pose is functionally the same
  for the project metric).

### Round 4 — 2026-06-19 (snapshot: `labels_raw_snapshot_2026-06-19_v4/`)

3,687 labels total (+296 since R3). First round with the **one-time
seed-queue** (750 neighbors of rare-class labels surfaced before random
sampling).

| Class | R3 | R4 | Δ | Notes |
|---|---:|---:|---:|---|
| unclear | 2,246 | 2,344 | +98 | many seeded crops turned out to be unclear |
| regular_facing_right | 217 | 217 | 0 | |
| regular_facing_left | 158 | 158 | 0 | |
| diag_up_left_facing_down | 163 | 163 | 0 | |
| head_up_facing_left | 124 | 139 | +15 | |
| diag_down_right_facing_down | 128 | 128 | 0 | |
| diag_down_left_facing_up | 29 | 91 | **+62** | crossed 30 |
| diag_up_right_facing_up | 10 | 77 | **+67** | crossed 30 (the seed's biggest win) |
| head_down_facing_right | 83 | 83 | 0 | |
| head_down_facing_left | 54 | 72 | +18 | |
| diag_up_right_facing_down | 62 | 62 | 0 | |
| upside_down_facing_right | 40 | 56 | +16 | |
| diag_down_left_facing_down | 41 | 41 | 0 | |
| upside_down_facing_left | 16 | 19 | +3 | seed didn't help much |
| head_up_facing_right | 18 | 18 | 0 | seed didn't help at all |
| diag_down_right_facing_up | 1 | 15 | +14 | from absent to weak |
| diag_up_left_facing_up | 1 | 4 | +3 | still essentially absent |

**Verdict on the one-time seed:**

- **Weak classes (<30): 6 → 4.** Two classes (`diag_down_left_facing_up`,
  `diag_up_right_facing_up`) crossed the usable threshold.
- **Best wins**: `diag_up_right_facing_up` (10→77, 7.7x) and
  `diag_down_left_facing_up` (29→91, 3x). These poses *do* exist in the
  videos — the seed just needed to point at them.
- **Two seeds underdelivered**: `head_up_facing_right` (no change, 180
  seeded) and `upside_down_facing_left` (+3, 160 seeded). The neighbors of
  these poses mostly turned out to be `unclear` or a different pose —
  i.e. the fish only stays in these positions for very brief moments.
- **Truly absent classes**: `diag_up_left_facing_up` (4 total) is essentially
  unusable. `diag_down_right_facing_up` (15) is weak but viable if we drop
  the per-class minimum.

**Recommended actions before training**:
1. Drop `diag_up_left_facing_up` entirely (4 samples — can't even split).
2. Keep `diag_down_right_facing_up`, `head_up_facing_right`,
   `upside_down_facing_left` but consider folding them with their less-rare
   mirror class (e.g. merge `*_left` and `*_right` for these specific cases
   to recover statistical power). The "% time upside-down" metric doesn't
   care about which side the fish is facing.

### Final — 2026-06-19 (snapshot: `labels_raw_snapshot_2026-06-19_final/`)

3,982 labels total. Labeling phase done.

| Class | Final | Tier |
|---|---:|---|
| unclear | 2,446 | reject bin |
| regular_facing_right | 230 | strong |
| regular_facing_left | 160 | strong |
| diag_up_left_facing_down | 176 | strong |
| head_up_facing_left | 150 | strong |
| diag_down_right_facing_down | 133 | strong |
| upside_down_facing_right | 121 | strong |
| diag_down_left_facing_up | 93 | strong |
| head_down_facing_right | 86 | strong |
| diag_up_right_facing_down | 80 | strong |
| diag_up_right_facing_up | 77 | strong |
| head_down_facing_left | 75 | strong |
| head_up_facing_right | 60 | strong |
| diag_down_left_facing_down | 49 | medium |
| diag_down_right_facing_up | 23 | weak |
| upside_down_facing_left | 19 | weak |
| diag_up_left_facing_up | 4 | unusable — drop |

**Counts vs round 1**: 16 usable classes (≥20 samples), up from 8 in R1.
Useful-label total: ~1,536, up from ~870 in R1 (+77%).

**Recommended cleanup before splitting:**
- Drop `diag_up_left_facing_up` (4) entirely with `--skip-class`.
- Either drop or merge-with-mirror `diag_down_right_facing_up` (23) and
  `upside_down_facing_left` (19). The mirror strategy makes the dataset more
  balanced (merged `upside_down` = 19+121 = 140 instead of two skewed bins).
- `unclear` continues to be excluded from reported percentages at inference.

### Round 6 — 2026-06-19 (snapshot: `labels_raw_snapshot_2026-06-19_v6/`)

4,269 labels total (+287 since "final"). General labeling, no targeted seed.

| Class | Final | R6 | Δ |
|---|---:|---:|---:|
| unclear | 2,446 | 2,553 | +107 |
| regular_facing_right | 230 | 265 | +35 |
| regular_facing_left | 160 | 191 | +31 |
| head_up_facing_left | 150 | 177 | +27 |
| diag_up_left_facing_down | 176 | 202 | +26 |
| diag_down_right_facing_down | 133 | 150 | +17 |
| head_down_facing_right | 86 | 102 | +16 |
| diag_up_right_facing_down | 80 | 90 | +10 |
| diag_down_left_facing_down | 49 | 54 | +5 |
| head_down_facing_left | 75 | 79 | +4 |
| upside_down_facing_right | 121 | 125 | +4 |
| head_up_facing_right | 60 | 63 | +3 |
| diag_up_right_facing_up | 77 | 78 | +1 |
| upside_down_facing_left | 19 | 20 | +1 |
| diag_down_left_facing_up | 93 | 93 | 0 |
| diag_down_right_facing_up | 23 | 23 | 0 |
| diag_up_left_facing_up | 4 | 4 | 0 |

Three classes still hard-stuck (`diag_up_left_facing_up` 4, `upside_down_facing_left` 20, `diag_down_right_facing_up` 23). These remain candidates for skip/merge during training experiments.

### Round 7 — 2026-06-20 (snapshot: `labels_raw_snapshot_2026-06-20/`)

4,722 labels total (+453 since R6).

| Class | R6 | R7 | Δ | Notes |
|---|---:|---:|---:|---|
| unclear | 2,553 | 2,762 | +209 | |
| diag_up_left_facing_down | 202 | 254 | +52 | |
| regular_facing_right | 265 | 297 | +32 | |
| upside_down_facing_left | 20 | 51 | **+31** | finally crossed 30 |
| regular_facing_left | 191 | 218 | +27 | |
| diag_down_right_facing_down | 150 | 172 | +22 | |
| head_up_facing_left | 177 | 198 | +21 | |
| head_down_facing_right | 102 | 123 | +21 | |
| diag_up_right_facing_down | 90 | 100 | +10 | |
| diag_down_left_facing_down | 54 | 63 | +9 | |
| head_down_facing_left | 79 | 88 | +9 | |
| head_up_facing_right | 63 | 68 | +5 | |
| upside_down_facing_right | 125 | 128 | +3 | |
| diag_up_right_facing_up | 78 | 80 | +2 | |
| diag_down_left_facing_up | 93 | 93 | 0 | |
| diag_down_right_facing_up | 23 | 23 | 0 | stuck |
| diag_up_left_facing_up | 4 | 4 | 0 | stuck |

Only **2 classes still <30** (`diag_up_left_facing_up` 4, `diag_down_right_facing_up` 23). `upside_down_facing_left` finally crossed.

## Parameters reference (for Phase 3)

A cheat sheet on what each knob actually does. Use it to pick what to sweep.

### Training duration & convergence

| Param | What it does | When to change it |
|---|---|---|
| `epochs` | Number of passes over the dataset | Too few → underfit (train loss still falling at end). Too many → overfit (val loss rising while train loss falls). Watch the loss curves; pick the elbow. Default 50 is a fine starting point. |
| `patience` | Early-stop if no val improvement for N epochs | Set to ~20% of epochs. Stops you wasting time once the model plateaus. |
| `batch` | Crops per gradient step | Smaller = noisier gradients (sometimes helps generalization), uses less memory. Larger = smoother but can overfit. On MPS, 32 is a sensible default. |

### Optimization

| Param | What it does | When to change it |
|---|---|---|
| `lr0` | Initial learning rate | Higher = faster but risks divergence. Lower = stable but slow. For fine-tuning a pretrained model, smaller (1e-3 → 1e-4) usually wins. |
| `lrf` | Final-to-initial LR ratio | Cosine schedule decays from `lr0` to `lr0 * lrf`. 0.01 means decay to 1% of starting LR. |
| `momentum` | SGD momentum | Smooths gradients. 0.9-0.95 is standard. Rarely worth tweaking. |
| `weight_decay` | L2 regularization | Bigger = stronger penalty on large weights = less overfitting. Try 1e-4 to 1e-3. |
| `optimizer` | SGD / Adam / AdamW | Adam/AdamW converges faster and is more forgiving. SGD with momentum often generalizes slightly better. |

### Data / augmentation

| Param | What it does | When to change it |
|---|---|---|
| `imgsz` | Crop is resized to this square before training | Larger = more detail visible, slower. 224 is standard, 320 captures more for high-res crops. |
| `degrees` | Max rotation augmentation | 0 = none. Small (5-15°) good for camera-tilt robustness. Anything that crosses a class boundary is wrong (see Phase 2 note). |
| `translate` | Random translation as fraction of image size | 0.1 = ±10%. Keeps subject roughly centered. |
| `scale` | Random scale augmentation | 0.5 = scale up to 1.5x or down to 0.5x. Helps the model handle distance variation. |
| `hsv_h, hsv_s, hsv_v` | Hue / saturation / value jitter | Helps with lighting/color robustness. 0.015 / 0.7 / 0.4 are ultralytics defaults. |
| `fliplr, flipud` | Probability of horizontal/vertical flip | **Both must stay at 0** for direction-aware classes. Either > 0 will change the correct label. |
| `mosaic` | Probability of mosaic augmentation (4 crops stitched) | Improves robustness but can hurt simple classification. Try 0 vs 1. |
| `erasing` | Probability of random pixel-block erasing | Helps the model not rely on a single visual feature. Try 0.0 vs 0.4. |

### Validation / data splits

| Param | What it does | When to change it |
|---|---|---|
| `val_ratio` | Fraction of labeled crops held out for validation | Smaller (0.1) = more train data, but val accuracy is noisier. Larger (0.25) = stable val signal, less train data. With <500 labeled crops, 0.15-0.2 is the sweet spot. |
| `test_ratio` | Fraction held out and never seen during training | Keep this fixed (0.1-0.15) across runs so reported test accuracies are comparable. |
| `seed` | RNG seed for the split | Fixing this makes runs comparable. Varying it (e.g. 5 different seeds) is how you measure how much of your "accuracy" is just split luck. |

### Inference-time knobs

| Param | What it does | When to change it |
|---|---|---|
| Confidence threshold | If `top1` probability < threshold, treat as "no prediction" | Tightening this trades coverage for precision. Useful if you want to route low-confidence frames to `unclear` instead of forcing a class. |
| Temperature | Divides the logits before softmax | T>1 makes the distribution flatter (model less confident); T<1 sharpens it. Usually used during *calibration*, not training. Useful if your top-1 probabilities are wildly overconfident on small classes. |
| Smoothing window | (Inference) # of neighboring frames used in majority vote in `infer.py` | Bigger window = smoother class trajectory, but blurs short transitions. 5 is a fine default; try 3, 7, 11. |

### Suggested first sweep (4 configs)

Start with these to learn what moves the needle on *your* data:

1. **`yolo_n_default`** — baseline, no augmentation beyond ultralytics defaults, 50 epochs.
2. **`yolo_n_continuous_aug`** — adds `degrees=10`, `erasing=0.4`. Same data.
3. **`yolo_n_cardinal_aug`** — uses cardinal rotation with class remapping. Same other settings as default.
4. **`yolo_s_default`** — same as #1 but uses `yolo11s-cls.pt` (bigger backbone) instead of `yolo11n-cls.pt`. Tells you whether nano is bottlenecking.

Plus the LR baseline:

5. **`logreg_raw_pixels`** — sklearn LogisticRegression on flattened resized pixels. Sets the floor.

After running these you'll know: does augmentation help? does more data via remapping help? is the backbone the limit? does the task even need a CNN?


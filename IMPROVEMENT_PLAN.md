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


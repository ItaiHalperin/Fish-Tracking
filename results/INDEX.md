# Training run index

One row per training run. After running `python -m classifier.train --config <c>`
and `python -m classifier.test --run results/<dir>`, append a row here by hand.

| Run dir | Model | Aug notes | Train / Val / Test | Acc | Macro F1 | Notes |
|---|---|---|---|---|---|---|
| yolo_n_default_20260624_151209 | yolo11n-cls (YOLO baseline, off-the-shelf) | ultralytics defaults (randaugment), no flip | 1521 / 363 / 266 | 60.53% | 0.516 | Off-the-shelf YOLO (our test harness). Lowball: default randaugment corrupts direction labels. |
| yolo_n_default_20260627_140856 | yolo11n-cls (YOLO baseline, fair) | auto_augment OFF, no flip | 1521 / 363 / 266 | 66.54% | 0.582 | Fair YOLO (no direction-corrupting aug, matching our models). +6pt over off-the-shelf. Still well below ResNet cRT (83%). |
| multihead_default_20260619_113646 | multihead (resnet34) | flip+remap, deg10, erase0.3, CB-loss, wsampler | 972 / 333 / 231 | 79.65% | 0.762 | First factorized run (split: old `labels`, now lost). Pose head strong (6 pose-only errs); facing weaker (down↔l/r). 28/47 errors both heads wrong = ambiguous crops. |
| multihead_reg_20260619_171417 | multihead (resnet34) | + dropout0.3, mixup0.2, ls0.05, wd5e-4, freeze5 | 972 / 333 / — | — | val 0.769 | Over-regularized: val macroF1 < baseline 0.799, train loss stuck ~1.1 (underfit). Test split lost (webapp overwrote `labels/`). Superseded by labels_split + moderated config. |
| multihead_default_20260619_232532 | multihead (resnet34) | flip+remap, deg10, erase0.3, CB-loss, wsampler | 1139 / 375 / 268 | 80.60% | 0.838 | New baseline on `labels_split` (more data). Macro F1 +0.076 vs first run — extra data helped the tail. |
| multihead_reg_20260620_083559 | multihead (resnet34) | + dropout0.2, ls0.05, mixup0.1, wd2e-4, freeze3, TTA | 1139 / 375 / 268 | 82.84% | 0.834 | Moderate reg + TTA. Acc +2.2pt but macro F1 tied (−0.004). Confirms data/label ceiling, not a tuning ceiling. |
| multihead_decoupled_20260620_152123 | multihead (resnet34) | Option 2: cRT two-stage (s1 natural+CE, s2 frozen heads balanced+CB), TTA | 1139 / 375 / 268 | 87.31% | 0.873 | **Current best.** +0.035 macro F1 vs baseline — decoupling fixed the classifier's tail bias. Weakest real classes now ~0.86 F1. |
| multihead_decoupled_20260620_175813 | multihead (resnet34) | Option 2 cRT, post-mining data, class-aware split | 1521 / 363 / 266 | 83.08% | 0.718 | After 2 mining rounds + class-aware split fix. NOT comparable to 0.873 (different split, test now includes harder rare classes — 14 vs 13). Macro F1 dragged by thin-support tail classes in test. |
| roll_cls_20260620_190833 | roll_cls (resnet34) | **ROLL label space (4-way, not composite)** | 1521 / 363 / 266 | 91.73% | 0.861 | Roll-only, no new labels. Upside-down acc 98.1%; belly_up F1 0.97, belly_down 0.94. Flanks weaker (right_flank F1 0.69 n=12, left_flank 0.85). Roll-only is strong — the actual goal. |
| angle_reg_20260620_210454 | angle_reg (resnet34) | Option 3: heading (continuous, 2150 line labels) + roll regression | 1521 / 363 / 266 | 80.08% | 0.699 | Composite snapped acc (col). **Roll acc 92.11% / macroF1 0.852 / upside 98.5% — TIED with roll_cls → heading doesn't help roll.** Heading itself accurate: 7.9° mean err. |
| roll_cls_angular_20260821_225029 | roll_cls (resnet34) | **distance-aware loss** (tau=45°), else identical to roll_cls | 1521 / 363 / 266 | **93.61%** | **0.903** | **Best roll model.** Ablation of roll_cls: only the loss differs. +1.88 acc / **+0.042 macro F1**, roll err 8.8°→6.8°. Gain is in the flanks (right_flank F1 0.69→0.83). Upside-down 98.12%→97.74% (one crop; noise). |
| multihead_angular_20260821_230731 | multihead (resnet34) | cRT + **distance-aware joint loss** (tau=45°), else identical to multihead_decoupled | 1521 / 363 / 266 | 82.71% | 0.711 | Ablation of multihead_decoupled. Wash on classes (−0.38 acc, −0.007 F1) but heading err 14.2°→13.3° — the excess over the 12.1° oracle floor halved (2.1°→1.2°). Confirms the composite task was already near its taxonomy limit. |

### Angular error — the same runs scored in degrees (split 1521 / 363 / 266)

Class accuracy treats a 45°-adjacent miss and a 180° miss as the same event. These
rows re-score the runs above by *how far off* the orientation was. Produced by
`python -m classifier.test --run results/<run>` (no retraining); heading truth is
the 2,150 continuous line annotations, roll truth is the class centre.

| Run | Class acc | Heading mean / median | ≤15° | ≤30° | Roll mean | Upside-down (±45°) |
|---|---|---|---|---|---|---|
| yolo_n_default_20260627_140856 | 66.5% | 22.4° / 12.5° | 60% | 89% | 32.8° | 85.71% |
| multihead_decoupled_20260620_175813 | 83.1% | 14.2° / 11.5° | 61% | 92% | 9.1° | 98.12% |
| angle_reg_20260620_210454 | 80.1% | **8.4° / 5.6°** | **85%** | **98%** | 11.2° | 98.50% |
| multihead_angular_20260821_230731 | 82.7% | 13.3° / 11.2° | 61% | 93% | 8.8° | 98.50% |
| **oracle floor** (perfect class → centre) | 100% | *12.1° / 11.0°* | — | — | — | — |

**The oracle floor is the headline.** It is the heading error a *perfect*
classifier still pays, because a classifier can only ever emit a bin centre. Two
consequences:

- **The classifier-vs-regression verdict is metric-dependent.** cRT wins on class
  accuracy (83.1% vs 80.1%) but loses badly on angular error (14.2° vs 8.4°; 61%
  vs 85% within 15°). The regressor beats the floor that bounds every classifier
  on this taxonomy.
- **cRT is nearly at the taxonomy's information limit** — 14.2° against a 12.1°
  floor, so only ~2° of its heading error is actual misclassification. Little
  headroom remains in the composite classifier.

Roll runs the other way (`roll_test_report.json` → `roll_angle`): roll_cls averages
**8.8°** vs the regressor's 11.2°, despite the near-tie on accuracy — the
regressor's right-flank error averages 59.2° (n=12) vs 15.0°. Same accuracy,
bigger misses. roll_cls's belly_up error is 2.5°, i.e. the project's actual metric
is essentially solved.

### Distance-aware loss (roll ablation, in degrees)

| Run | Roll acc | Macro F1 | Roll mean err | right_flank F1 (n=12) |
|---|---|---|---|---|
| roll_cls_20260620_190833 | 91.73% | 0.861 | 8.8° | 0.69 |
| roll_cls_angular_20260821_225029 | **93.61%** | **0.903** | **6.8°** | **0.83** |

Also found while measuring: the factorized model emitted **10/266 (3.8%)
impossible pose×facing combinations** — independent per-head argmax can compose
classes the 16-class taxonomy never defines (`head_up_facing_up`). `predict_joint`
now restricts the argmax to composites that exist.

### Phase 4 baselines (floor, split 1521 / 363 / 266)

| Baseline | Task | Acc | Macro F1 |
|---|---|---|---|
| Majority class | composite (16-way) | 12.0% | 0.015 |
| Logistic reg. on raw pixels | composite (16-way) | 54.9% | 0.494 |
| Majority class | roll (4-way) | 51.5% | 0.170 |
| Logistic reg. on raw pixels | roll (4-way) | 63.2% | 0.530 |

Best models beat LR-pixels by ~28 acc points on both tasks; majority macro-F1 ≈0.02–0.17 confirms the task is non-trivial.

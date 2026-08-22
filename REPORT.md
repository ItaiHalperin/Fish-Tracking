# Fish Orientation Recognition — Technical Report

## 1. Problem
Given side-view aquarium video with per-fish tracking, classify each fish crop's
orientation. Two targets: **roll** (rotation about the body's long axis —
upright / on-side / upside-down), the primary goal, and the **full position**, a
16-way composite of body **pose** (regular, diagonal, head-up/down, upside-down)
× head-**facing** direction. Crops are named `video__id_N_frame_M.jpg`.

The central difficulty is **severe class imbalance**: the composite labels form a
long-tailed distribution (head class ≈217 crops, several tail classes 1–30, some
empty), which biases naïve classifiers toward majority classes [1].

## 2. Data and labeling
We built a web labeler and grew the set to **~2,150 crops / 16 classes** (plus a
~2,650 "unclear" reject bucket, excluded). Labeling used **model-assisted active
learning** [7]: after each model, we mined the unlabeled pool for likely
tail-class crops using (i) backbone-feature nearest-neighbor retrieval to labeled
rare examples, (ii) tail-class predictions, and (iii) prediction uncertainty,
then verified each by hand. This raised the rarest classes (e.g. 4 → 28 crops)
until genuinely scarce poses hit diminishing returns. A separate line-annotation
tool recorded a **continuous head-direction angle** for all crops.

Splits are **leak-free and class-aware**: assignment is by fish *track* (a fish's
near-duplicate frames never straddle train/test), placed rarest-class-first with a
guarantee that every class appears in training — an instance of group-aware,
stratified splitting. A naïve random split had put an entire 28-crop class
wholly in the test set (0 in train), which alone dropped macro-F1 ~0.15; the
class-aware split fixed this.

## 3. Methods (with background)
All models use a pretrained ResNet-34 backbone [8].

- **Factorized multi-head.** Rather than one 16-way softmax, two heads predict
  pose and facing independently, reducing a 217:1 / 16-class problem to ~7:1 /
  8-class (pose) + ~10:1 / 4-class (facing). This also makes horizontal-flip
  augmentation usable (it maps left↔right consistently in both factors).
- **Imbalance handling.** Class-balanced loss via the effective-number
  reweighting of Cui et al. [3], optional focal loss [4], and inverse-frequency
  weighted sampling.
- **Regularization.** Dropout, mixup [5], label smoothing [6], and test-time
  augmentation.
- **Decoupled two-stage training (cRT)** [2]. Stage 1 learns the representation
  on the *natural* distribution with plain cross-entropy; stage 2 freezes the
  backbone, re-initializes the heads, and retrains only them on *class-balanced*
  sampling. Kang et al. show rebalanced sampling harms representation learning but
  helps the classifier — so the two are best decoupled.
- **Angle regression.** Orientation as continuous heading + roll, each encoded as
  a (cos, sin) vector with an angular loss to handle the 360° wrap-around — the
  Biternion-Net approach to regressing continuous angles from (initially discrete)
  labels [9]. Class→angle targets come from a fixed mapping; heading was later
  refined with the continuous line annotations.
- **Roll-only model** and a **heading-as-auxiliary** test, motivated by
  multi-task learning theory [10]: does jointly predicting heading improve roll?
- **Distance-aware (cost-sensitive) loss.** Cross-entropy charges every wrong
  class equally, which for an orientation task is plainly wrong: mistaking a pose
  for its 45° neighbour is not the error that mistaking it for the 180° opposite
  is. We keep the labels as plain classes and replace the one-hot target with
  `softmax(-d(i, ·)/τ)`, where `d` is angular distance on the circle in both
  orientation axes (from the class→angle map) and τ is in degrees — the
  label-relaxation / cost-sensitive-learning family [11], and the discrete
  counterpart of the angular loss used by the regressor. τ→0 recovers ordinary
  weighted cross-entropy exactly, so each run is a clean one-knob ablation.
  For the factorized model the term must act on the *composite*: pose×facing is a
  different factorization from heading×roll (`facing_right` is a 0° heading under
  a `regular` pose but a 90° roll under `head_up`), so an angular cost is only
  well-defined once the two factors are combined.

## 4. Results
All rows share one split per block. Composite = 16-way full position; Roll =
4-way roll. (Macro-F1 weights every class equally, so it reflects tail
performance.)

**Baselines (floor, same split 1521 / 363 / 266).** Majority-class and
multinomial logistic regression on raw 32×32 pixels:

| Task | Majority | LR-pixels | Best model |
|---|---|---|---|
| Composite (16-way) | 12.0% / 0.015 | 54.9% / 0.494 | **83.1% / 0.718** (cRT) |
| Roll (4-way) | 51.5% / 0.170 | 63.2% / 0.530 | **91.7% / 0.861** (roll-only) |

The task is non-trivial (majority macro-F1 ≈0.02–0.17), pixels alone are a
meaningful but limited floor, and the learned models beat LR-pixels by ~28
accuracy points on both tasks — the representation, not the label distribution,
drives the result.

**Full-position classifier (split 1139 / 375 / 268):**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Factorized multi-head (baseline) | 80.6% | 0.838 |
| + moderate regularization + TTA | 82.8% | 0.834 |
| **+ decoupled cRT** | **87.3%** | **0.873** |

Regularization did not improve macro-F1 (the model was data-limited, not
over-fit); decoupling gave +0.035 macro-F1, the largest single gain.

**After 2 mining rounds, on the larger class-aware split (1521 / 363 / 266):**

| Model | Task | Accuracy | Macro-F1 | Extra |
|---|---|---|---|---|
| Decoupled cRT | Composite | 83.1% | 0.718 | harder test set¹ |
| Angle regression | Composite (snapped) | 80.1% | 0.699 | heading err **7.9°** |
| **Roll-only classifier** | **Roll (4-way)** | **91.7%** | **0.861** | **upside-down 98.1%** |
| Angle-reg roll head | Roll (4-way) | 92.1% | 0.852 | upside-down 98.5% |

¹ Not comparable to the 0.873 above: the rebuilt split's test set includes more
thin-support tail classes (14 vs 13 tested), which macro-F1 penalizes.

**Key comparisons.**
- *Classes:* the **decoupled classifier beats angle-regression** on the discrete
  task (83.1/0.718 vs 80.1/0.699, same split) — a model trained for the discrete
  boundaries beats one adapted from continuous angles, where snapping compounds
  two errors. But see §4a: this verdict is specific to the discrete metric and
  reverses when the same runs are scored in degrees.
- *Roll:* **roll-only ≈ joint** (91.7/0.861 vs 92.1/0.852) — adding the heading
  head does **not** improve roll, so heading is unnecessary for the primary goal.
- Heading itself is accurate (7.9° mean error) if a continuous output is wanted.

## 4a. Scoring in degrees, not just right/wrong

Accuracy and macro-F1 treat a 45°-adjacent miss and a 180° miss as the same
event, which hides most of what an orientation model gets wrong. Re-scoring the
*same trained runs* by angular error (heading truth = the 2,150 continuous line
annotations; no retraining) gives a different and sharper picture.

The reference quantity is the **oracle floor**: the heading error a *perfect*
classifier still pays, because a classifier can only ever emit a bin centre. On
this taxonomy it is **12.1° mean / 11.0° median**.

| Model | Class acc | Heading mean / median | ≤15° | Roll mean | Upside-down (±45°) |
|---|---|---|---|---|---|
| YOLO baseline (fair) | 66.5% | 22.4° / 12.5° | 60% | 32.8° | 85.7% |
| Decoupled cRT | 83.1% | 14.2° / 11.5° | 61% | 9.1° | 98.1% |
| Angle regression | 80.1% | **8.4° / 5.6°** | **85%** | 11.2° | 98.5% |
| *oracle floor* | — | *12.1° / 11.0°* | — | — | — |

Three results follow.

1. **The classifier-vs-regression verdict is metric-dependent.** cRT wins on class
   accuracy (83.1% vs 80.1%) and loses on angular error (14.2° vs 8.4°; 61% vs 85%
   within 15°). The regressor is *below the floor that bounds every classifier on
   this taxonomy* — it cannot be matched by any model restricted to these 16 bins.
   Both statements were always true; they had only ever been measured on one axis.
2. **The composite classifier is near the information limit of its label space.**
   14.2° against a 12.1° floor leaves ~2.1° attributable to misclassification.
   Most of the residual "17% inaccuracy" is the taxonomy, not the model.
3. **Roll reverses the ordering.** roll-only averages 8.8° against the regressor's
   11.2° despite the near-tie on accuracy: the regressor's right-flank error
   averages 59.2° (n=12) vs 15.0°. Equal accuracy, larger misses. roll-only's
   belly_up error is 2.5°, so the project's actual metric is essentially solved.

Incidentally, the factorized model emitted **10/266 (3.8%) impossible pose×facing
combinations** — taking each head's argmax independently can compose classes the
16-class taxonomy never defines (`head_up_facing_up`). Restricting the argmax to
composites that exist (`predict_joint`) fixes this.

## 4b. Distance-aware loss

Same configs as the two best models, changing only the loss (τ = 45°).

| Task | Model | Acc | Macro-F1 | Angular error |
|---|---|---|---|---|
| Roll (4-way) | roll-only, CE | 91.7% | 0.861 | 8.8° |
| Roll (4-way) | **roll-only, distance-aware** | **93.6%** | **0.903** | **6.8°** |
| Composite | cRT, CE | 83.1% | 0.718 | 14.2° heading |
| Composite | cRT, distance-aware | 82.7% | 0.711 | 13.3° heading |

**On roll it is the largest single gain in the project bar factorization**:
+0.042 macro-F1 and −2.0° angular error, from one loss term and no new labels
(comparable to the +0.035 that decoupled cRT bought on the composite task). The
gain is concentrated in the flanks — `right_flank`, the weakest class in the
project, goes from F1 0.69 to 0.83 — which is exactly the mechanism working as
intended: the soft target encodes that a flank is *nearer* to belly_down than
belly_up is, an ordering flat cross-entropy cannot express. Against this,
upside-down accuracy moved 98.12% → 97.74%, a single crop of 266 — noise, but
noted because it is the headline metric.

**On the composite task it is a wash** (−0.38% accuracy is one crop), with a small
gain in degrees. That small gain is the more informative number: since no
classifier can beat the 12.1° floor, only 2.1° of the baseline's 14.2° was ever
recoverable, and this run recovered nearly half of it (excess 2.1° → 1.2°). §4a
predicted precisely this — the composite task had almost no headroom left, so a
better-shaped loss has little to win there.

## 5. Conclusions
- **Roll/upside-down:** the roll-only classifier with the **distance-aware loss**
  is the deliverable (93.6% 4-way, 0.903 macro-F1, 6.8° mean roll error,
  97.7% upside-down).
- **Full positions:** the decoupled-cRT classifier (best discrete accuracy and
  native class output); the distance-aware variant is equivalent on classes and
  slightly better in degrees.
- **Continuous orientation:** the angle-regression model — and on angular error it
  beats every classifier here, including a hypothetically perfect one, so prefer
  it whenever the output is an angle rather than a class name.
- Biggest levers were **label factorization + decoupled training** (for classes),
  **task scoping to roll** (for the goal), and **charging mistakes by their
  angular size** (for roll quality). The binding constraint on the composite task
  is now the **taxonomy itself** — 12.1° of the 14.2° heading error is the cost of
  discretizing into 16 bins, not model error — with **data** for genuinely rare
  orientations the constraint behind it. Neither is a capacity problem.
- **Methodological lesson:** the choice of metric decided a conclusion. Measuring
  "how wrong" rather than "whether wrong" reversed the classifier-vs-regression
  comparison and showed the composite classifier had almost no headroom left.
  Report both for any task whose labels are a discretized continuum.

## References
1. Zhang et al. *Deep Long-Tailed Learning: A Survey.* IEEE TPAMI, 2023. arXiv:2110.04596.
2. Kang et al. *Decoupling Representation and Classifier for Long-Tailed Recognition.* ICLR 2020. arXiv:1910.09217.
3. Cui et al. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019. arXiv:1901.05555.
4. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017. arXiv:1708.02002.
5. Zhang et al. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018. arXiv:1710.09412.
6. Müller et al. *When Does Label Smoothing Help?* NeurIPS 2019. arXiv:1906.02629.
7. Settles. *Active Learning Literature Survey.* Univ. Wisconsin–Madison TR 1648, 2009.
8. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385.
9. Beyer et al. *Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels.* GCPR 2015.
10. Ruder. *An Overview of Multi-Task Learning in Deep Neural Networks.* 2017. arXiv:1706.05098.
11. Diaz & Marathe. *Soft Labels for Ordinal Regression.* CVPR 2019. (Distance-weighted soft targets for ordered/circular label spaces; see also Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI 2001.)

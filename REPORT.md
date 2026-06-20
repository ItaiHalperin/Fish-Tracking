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
  two errors.
- *Roll:* **roll-only ≈ joint** (91.7/0.861 vs 92.1/0.852) — adding the heading
  head does **not** improve roll, so heading is unnecessary for the primary goal.
- Heading itself is accurate (7.9° mean error) if a continuous output is wanted.

## 5. Conclusions
- **Roll/upside-down:** the roll-only classifier is the deliverable (91.7% 4-way,
  98.1% upside-down).
- **Full positions:** the decoupled-cRT classifier (best discrete accuracy and
  native class output).
- **Continuous orientation:** the angle-regression model, when exact angles or
  flexible re-binning are needed.
- Biggest levers were **label factorization + decoupled training** (for classes)
  and **task scoping to roll** (for the goal); the biggest constraint was **data**
  for genuinely rare orientations, not model capacity.

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

# Handling misrepresented classes — research notes

Companion to [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md), Phase 2 (augmentation) and
beyond. Focus: what to do about the very uneven class distribution.

## Your actual distribution (from `labels_raw_v1_archive`)

```
159  unclear              22  diag_up_right_facing_up
120  diag_down            14  diag_down_left_facing_up
119  regular              11  top_diagonal
114  head_up              10  diag_up_right_facing_down
102  diag_down_right_..    8  diag_down_left_facing_down
 85  head_up_facing_left   7  regular_facing_left
 84  diag_up               7  head_up_facing_right
 56  head_down             5  diag_up_left_facing_down
 36  top_view              2  top_view_facing_left
 31  regular_facing_right  2  top_view_facing_down
 26  head_down_facing_left 1  diag_down_right_facing_up
                           0  (≥8 classes with zero samples)
```

This is a **severe long-tailed distribution**: head ~120–160, tail at 0–2, plus
several classes that have **no examples at all**. That last fact matters a lot —
see "Read this first".

---

## Read this first — the biggest lever isn't a loss function

For your data the dominant problem is **taxonomy fragmentation**, not the
training algorithm. You've defined ~30 *composite* classes (pose × facing
direction, e.g. `diag_up_right_facing_down`). With only ~250 labeled crops,
that splinters tiny amounts of data across many buckets, which is *why* the tail
is so thin. No re-weighting trick rescues a class with 0–2 examples.

Three concrete consequences:

1. **A class with 0 samples cannot be learned.** Reweighting, focal loss,
   oversampling — none of them create signal from nothing. Such classes must be
   *dropped, merged into a coarser parent, or actively labeled* before any
   algorithmic fix is meaningful.
2. **Factorize the label.** Instead of one 30-way head, predict pose (`regular`,
   `diag_up`, `head_down`, `top_view`, …) and facing direction as **separate
   outputs** (multi-head / multi-task). Each sub-problem then has far more
   examples per class and a much milder imbalance. This is the single highest-
   leverage change available to you.
3. **This is the real argument for Phase 5 (angle regression).** The fish-pose
   literature consistently prefers *regressing* a continuous orientation over
   classifying discretized bins, precisely because binning fragments data and
   forces mid-rotation fish into the wrong bucket. Your Phase 5 note already
   senses this; the research backs it. Treat "park Phase 5" as provisional — if
   the per-class recall on orientation stays bad after re-balancing, regression
   is the principled fix, not a luxury.

Do the taxonomy work first. Then the techniques below are what squeeze the rest.

---

## Technique catalogue (ordered by effort/payoff for your case)

The literature groups solutions into three families: **class re-balancing**,
**information augmentation**, and **module improvement** (decoupling/ensembles).
Below, mapped to your YOLO-cls + small-data setup.

### 1. Fix evaluation before anything else (free, do immediately)
Overall accuracy is misleading under imbalance — a model that ignores every tail
class still scores high. Track **macro-F1, per-class recall, and balanced
accuracy**, plus the confusion matrix. Your plan already writes a confusion
matrix and macro-F1 into `test_report.json` — make macro-F1 / per-class recall
the *primary* number you compare configs on, not top-1 accuracy.

### 2. Weighted / re-sampling dataloader (low effort, native to your stack)
Sample minority crops more often (probability ∝ inverse class frequency). This
is directly doable in Ultralytics by subclassing the classification dataset and
overriding sampling — see the YOLO tutorial link below. Cheap, uses all data, no
undersampling. Caveat: pure oversampling of a 2-image class just shows those 2
images many times → overfitting; pair it with strong augmentation (#4).

### 3. Cost-sensitive / class-balanced loss (low–medium effort)
Reweight the loss per class. Two standard forms:
- **Class-balanced loss** (weight by "effective number of samples", not raw
  inverse frequency — avoids over-boosting the 1–2-sample classes).
- **Focal loss** — down-weights easy, well-classified examples so training
  focuses on hard/rare ones. Originally for detection; usable for classification.
- **Logit adjustment** — add the log class-prior as a bias to the logits.
  Cheap, theoretically grounded, often matches fancier methods.

### 4. Augmentation aimed at the tail (medium effort — your Phase 2)
Your plan's safe knobs (rotation ≤15°, translate/scale, HSV, erasing) are
correct. Two additions for imbalance specifically:
- **mixup / CutMix** — convex combinations of images/labels. This is the
  image-domain analogue of SMOTE (classic SMOTE is for tabular data, *not*
  pixels — don't use it directly on crops). Helps regularize tail classes.
- **Class-conditional augmentation strength** — apply heavier augmentation to
  minority classes than to head classes, so a rare crop yields more varied views.
- Keep your direction-aware caps (no flips, no large rotations) — but note
  factorizing the label (#2 above) *removes* that constraint for the pose head,
  letting you use flips/rotations freely there.

### 5. Two-stage decoupled training (medium effort, strong payoff)
Decoupling representation learning from the classifier (a.k.a. cRT / LWS):
**Stage 1** train the backbone on the natural (imbalanced) distribution;
**Stage 2** freeze features and retrain *only* the classifier head on a balanced
sample (or with adjusted logits). Consistently one of the best-performing,
simplest "module improvement" recipes in the long-tail surveys.

### 6. Hierarchical / coarse-to-fine (medium effort — fits your label structure)
Your labels are naturally hierarchical (coarse pose → fine facing direction).
Train a coarse classifier first (well-populated, balanced-ish), then per-parent
fine classifiers only where enough data exists. Coarse levels are inherently
more balanced and the structure stops rare-class errors from cascading. This
pairs naturally with the factorization idea in #2.

### 7. Transfer / few-shot for the genuine tail (higher effort)
For classes you can't grow past a handful, lean on the pretrained backbone and
few-shot/metric-learning ideas (embed crops, classify by nearest class
prototype). Useful as a *baseline for the tail* even if the main model stays
softmax — and complements your Phase 4 "LR on embeddings" baseline.

---

## Recommended path for *your* project

1. **Re-taxonomize** (Phase 1-adjacent): drop/merge the 0–2-sample classes;
   split the label into separate *pose* and *facing-direction* heads. (Biggest win.)
2. **Switch the headline metric** to macro-F1 / per-class recall. (Free.)
3. **Add weighted sampling + class-balanced (or focal) loss** as config knobs
   alongside your Phase 2 augmentation. Sweep them like any other param (Phase 3).
4. **Add mixup + class-conditional aug strength** to the augmentation block.
5. **Try decoupled two-stage training** once #1–#4 are in — likely your best
   single accuracy jump on the tail.
6. **Revisit angle regression (Phase 5)** if orientation recall is still poor —
   the literature says it's the right model for continuous orientation, not a
   detour. Label more data regardless; algorithms can't manufacture rare-class signal.

---

## Links to read

### Surveys / overviews (start here)
- [Deep Long-Tailed Learning: A Survey (arXiv 2110.04596)](https://arxiv.org/abs/2110.04596) — the canonical map of the whole field (re-balancing / augmentation / module improvement).
- [A Survey of Deep Long-Tail Classification Advancements (arXiv 2404.15593)](https://arxiv.org/abs/2404.15593) — more recent, covers newer methods.
- [Low-shot learning and class imbalance: a survey (J. Big Data)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00851-z) — ties together few-shot + imbalance, relevant to your tiny tail classes.

### Loss functions / re-weighting
- [Focal Loss — Ultralytics glossary](https://www.ultralytics.com/glossary/focal-loss) — concise, in your framework's terms.
- [Focal Loss explained (Towards Data Science)](https://towardsdatascience.com/a-loss-function-suitable-for-class-imbalanced-data-focal-loss-af1702d75d75/) — intuition + math.
- [Class-Balanced Loss functions, empirical study (arXiv 2407.14381)](https://arxiv.org/pdf/2407.14381) — when reweighting helps vs hurts.

### YOLO / Ultralytics-specific (your stack)
- [Balance Classes During YOLO Training Using a Weighted Dataloader (y-t-g)](https://y-t-g.github.io/tutorials/yolo-class-balancing/) — concrete code: subclass dataset, inverse-frequency sampling. Most directly actionable.
- [Ultralytics issue #16205 — class imbalance proposed solution](https://github.com/ultralytics/ultralytics/issues/16205) — weighted classification loss from label counts.
- [Ultralytics forum — tools for handling class imbalance](https://community.ultralytics.com/t/tools-for-handling-class-imbalance/1149)
- [yolov5 discussion #5604 — assigning class weights](https://github.com/ultralytics/yolov5/discussions/5604)

### Augmentation / synthetic minority data
- [SMOTE for Imbalanced Classification (MachineLearningMastery)](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) — understand why classic SMOTE is tabular-only.
- [From SMOTE to Mixup for Deep Imbalanced Classification (arXiv 2308.15457)](https://arxiv.org/pdf/2308.15457) — the image-domain replacement for SMOTE.
- [DeepSMOTE: Fusing Deep Learning and SMOTE (arXiv 2105.02340)](https://arxiv.org/pdf/2105.02340) — generative oversampling for images, if you go heavier.
- [Oversampling and undersampling (Wikipedia)](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) — quick reference.

### Two-stage / hierarchical (module improvement)
- [MetaBalance: High-Performance Networks for Class-Imbalanced Data (arXiv 2106.09643)](https://arxiv.org/pdf/2106.09643) — practical two-stage-style recipe.
- [Hierarchical Classification Models (overview)](https://www.emergentmind.com/topics/hierarchical-classification-models) — coarse-to-fine, local-classifier-per-node, matches your label tree.
- [Deep Imbalanced Regression via Hierarchical Classification Adjustment (arXiv 2310.17154)](https://arxiv.org/abs/2310.17154) — bridges classification ↔ regression; relevant to your Phase 5 decision.

### Orientation / pose (the "position identification" angle)
- [Feasibility Research on Fish Pose Estimation Based on Rotating Box Detection (MDPI Fishes)](https://www.mdpi.com/2410-3888/6/4/65) — fish-specific orientation.
- [Pose estimation-based visual perception for fish swimming (bioRxiv)](https://www.biorxiv.org/content/10.1101/2022.09.07.507033.full.pdf) — uses regression on pose vector rather than class bins (supports Phase 5).
- [DeepPoseKit — animal pose estimation toolkit (eLife)](https://elifesciences.org/articles/47994) — keypoint-based alternative to orientation classes.
- [Animal behavior analysis with deep learning: a survey (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0957417425019499) — broader context.

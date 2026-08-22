# Fish Orientation Classification — Project Summary

## Goal
Determine the orientation of tracked fish from side-view aquarium video. The
primary deliverable is **roll** (is the fish upright, on its side, or upside
down); a secondary deliverable is the **full position** — a 16-way label
combining body pose and head-facing direction.

## Data & labeling
- Source: side-view tank video → per-fish tracked crops
  (`video__id_N_frame_M.jpg`).
- Built a custom web labeler and grew the set from a few hundred crops to
  **~2,150 labeled crops** across 16 composite classes (plus a ~2,650 "unclear"
  reject bucket that we exclude from training).
- Labeling methods that mattered (not just clicking faster):
  - **Random sampling + context strip** — pick diverse frames and show the
    neighboring frames of the same fish, so the chosen crop is easy to judge.
  - **Rare-class neighborhood seeding** — bias the queue toward scarce poses.
  - **Model-assisted active learning ("mining")** — use the trained model to
    surface likely rare-class crops (feature nearest-neighbor retrieval +
    tail-class predictions + uncertainty), then a human verifies each. This
    lifted starved classes (e.g. one class went 4 → 28 crops).
  - **Heading line-annotator** — drew the continuous head-direction angle on all
    2,150 crops for the angle-regression experiments.
- **Leak-free, class-aware splits** — split by fish *track* (the same fish never
  appears in both train and test), and guarantee every class appears in training.

## What we tried
| Approach | Idea | Outcome |
|---|---|---|
| Factorized multi-head | Predict pose and facing-direction with two heads instead of one 16-way label | ✅ Core idea — turned a 217:1 / 16-class problem into 8-way + 4-way |
| Class-balanced loss + weighted sampling | Counter class imbalance | ✅ Standard, used in all models |
| Regularization (dropout, mixup, label-smoothing, TTA) | Reduce overfitting | ➖ Little gain once data grew — data, not overfitting, was the ceiling |
| Decoupled two-stage (cRT) | Learn features on natural data, then retrain only the classifier on balanced data | ✅ **Best full-position classifier** |
| Active-learning mining | Use the model to find rare classes to label | ✅ Lifted the tail; ➖ diminishing returns on genuinely rare poses |
| Angle regression | Predict continuous heading + roll instead of classes | ✅ Works, gives continuous angles; ➖ slightly worse at discrete classes |
| Roll-only model | Focus on the actual goal | ✅ **Best for roll** |
| Heading as auxiliary | Does also predicting heading help roll? | ❌ No — roll-only ties the joint model |
| Scoring in degrees | Measure *how wrong*, not just *whether wrong* | ✅ Reversed a recorded verdict; exposed the taxonomy's error floor |
| Distance-aware loss | Charge a mistake by its angular size, not a flat right/wrong | ✅ **+0.042 macro-F1 on roll** (the deliverable); a wash on composite |

## Headline results
- **Roll (the goal): 93.6% 4-way accuracy, 0.903 macro-F1, 97.7% upside-down
  detection**, with a mean roll error of 6.8° (roll-only + distance-aware loss).
- **Full-position classifier: macro-F1 improved 0.76 → 0.87** (factorized +
  decoupled cRT).
- **Continuous orientation: heading predicted to ~8° mean error.**
- **Key finding:** heading is *not* needed for roll — the simple roll-only model
  matches the joint model, so the simplest model is the deliverable.
- **Charging mistakes by their angular size** (a 45° miss costs a quarter of a
  180° miss) lifted roll macro-F1 by 0.042 with no new labels — most of it in the
  weakest class, `right_flank`, F1 0.69 → 0.83.
- **The taxonomy, not the model, is now the limit on full position.** A *perfect*
  16-way classifier would still be 12.1° off on heading, because it can only emit
  bin centres; our classifier is at 13.3°. Scoring in degrees also reversed the
  classifier-vs-regression verdict: the regressor's 8.4° beats every classifier,
  including a hypothetically perfect one, even though it scores 3 points lower on
  class accuracy.

### Vs. baselines (is the task/model actually good?)
| Task | Dumb majority | Logistic reg. on pixels | **Our model** |
|---|---|---|---|
| Full position (16-way) | 12% | 55% | **83%** |
| Roll (4-way) | 52% | 63% | **92%** |

The task isn't trivially easy (majority is far below our models), and the deep
models beat a simple pixel classifier by ~28 points — so the result comes from
learned features, not an easy dataset.

## What didn't work / lessons
- Heavy regularization didn't beat the baseline once we had more data.
- A naïve random split once put an entire rare class into the test set with zero
  training examples, cratering the score — fixed with a class-aware track split.
- Mining can't create genuinely rare orientations that barely occur in the
  footage.

## Deliverables
- **Roll / upside-down:** roll-only classifier with the distance-aware loss
  (`configs/roll_cls_angular.yaml`) — recommended.
- **Full positions (named classes):** decoupled cRT classifier.
- **Continuous angles (optional):** angle-regression model — and the better choice
  whenever the output wanted is an angle rather than a class name.

# archive/ — superseded models. None of these is the deliverable.

Every model here was replaced by something better. They are kept so the
experiments are reproducible and the progression is inspectable. **The models the
system actually serves live in [`ml_storage/`](../ml_storage/README.md).**

Each run directory holds the `config.yaml` it was trained from, `weights/best.pt`,
and its test reports — enough to re-run it with
`python -m classifier.test --run archive/classifier_experiments/<run>`.

## classifier_experiments/

18 runs of the current `classifier/` package (ResNet-34 backbones, plus YOLO11n-cls
baselines). Grouped by approach, best of each first:

| Run | Approach |
|---|---|
| `roll_cls_angular_20260821_225029` | roll-only + distance-aware loss — **superseded only by its registered copy in `ml_storage/`** |
| `roll_cls_20260620_190833` | roll-only, flat cross-entropy |
| `multihead_angular_20260821_230731` | factorized pose×facing, cRT + distance-aware loss |
| `multihead_decoupled_2026062*` (3) | factorized + decoupled cRT two-stage |
| `multihead_default_2026061*` (2) | factorized baseline |
| `multihead_reg_*` (2) | factorized + heavy/moderate regularization |
| `angle_reg_20260620_210454` | continuous heading + roll regression |
| `angle_reg_20260620_202832` | earlier angle-regression attempt (no checkpoint kept) |
| `ablation_aug_{on,off}_20260627_*` | augmentation ablation pair |
| `yolo_n_default_20260627_140856` | YOLO11n-cls baseline, direction-safe augmentation |
| `yolo_n_default_20260624_151209` | YOLO11n-cls baseline, ultralytics defaults |

`baselines.json` holds the majority-class and logistic-regression-on-pixels floors
for both label spaces, from `python -m classifier.baselines`.

## detector_experiments/

Two YOLO11**n** detector trainings — a separate, weaker lineage from the served
YOLO11s detector, not its ancestors. `fish_tracking_model-2` is the better of the
two (200 epochs requested, early-stopped at 192; mAP@50 0.881, mAP@50-95 0.556);
`fish_tracking_model` is a 10-epoch run kept as an underfitting reference
(mAP@50 0.101). The served detector reaches mAP@50 0.953 — see
[`ml_storage/README.md`](../ml_storage/README.md). `detector_error_analysis/` holds its false-positive and
false-negative crops from `detection/evaluate_errors.py`.

## legacy_6class_classifier/

The original approach, before the label space was factorized: a 6-class YOLO11n-cls
classifier trained on the flat `labels/train|val` layout. `code/` is the training,
testing and video-inference scripts that produced `runs/` — kept together because
the current `classifier/` package cannot load these checkpoints. Superseded
entirely; here for the record.

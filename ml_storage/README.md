# ml_storage/ — the models the system serves

`analysis/pipeline.py` and the analysis webapp resolve their models from this
registry, not from `archive/`. Weight resolution is in `core/ml_storage.py`:
`default.pt` if present, else the most recent `run_*/weights/best.pt`.

| Slot | Serving                             | Notes                                                |
|---|-------------------------------------|------------------------------------------------------|
| `model_registry/detection/goldfish_yolo/default.pt` | YOLO26**s** detector (9.47M params) | mAP@50 0.9399, mAP@50-95 0.6500, P 0.9272 / R 0.9141 |
| `model_registry/classifier/fish_position_classifier/run_20260822_190700` | `roll_cls_angular`                  | 4-way roll: 93.61% accuracy, 0.903 macro-F1          |

`run_20260623_202149` is the previous roll classifier (91.73% / 0.861), kept for
comparison. Because resolution picks the newest run, adding a run changes what the
pipeline serves — the reported percentages are sensitive to this, so fix one
version for a whole study and re-run everything if it changes.

The detector's metrics above are read from the checkpoint's own `train_metrics`
(200 epochs, 640px, batch 64, patience 50). It was imported rather than trained
in-repo, so run folder has been imported — and it is **not** a descendant of the
YOLO11n runs in `archive/detector_experiments/`, which are a different, weaker
lineage.

Register a new model with `python -m classifier.train --config <cfg> --register`,
or import external weights with `python -m core.import_weights`.

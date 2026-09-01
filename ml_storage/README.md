# ml_storage/ — the models the system serves

`analysis/pipeline.py` and the analysis webapp resolve their models from this
registry, not from `archive/`. Weight resolution is in `core/ml_storage.py`:
`default.pt` if present, else the most recent `run_*/weights/best.pt`.

| Slot | Serving | Notes |
|---|---|---|
| `model_registry/detection/goldfish_yolo/default.pt` | YOLO11n detector | mAP@50 0.881, mAP@50-95 0.556 |
| `model_registry/classifier/fish_position_classifier/run_20260822_190700` | `roll_cls_angular` | 4-way roll: 93.61% accuracy, 0.903 macro-F1 |

`run_20260623_202149` is the previous roll classifier (91.73% / 0.861), kept for
comparison. Because resolution picks the newest run, adding a run changes what the
pipeline serves — the reported percentages are sensitive to this, so fix one
version for a whole study and re-run everything if it changes.

Register a new model with `python -m classifier.train --config <cfg> --register`,
or import external weights with `python tools/import_weights.py`.

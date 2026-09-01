# classifier/

Fine-tunes and runs a fish position classifier.

## Layout

```
classifier/
  config.py        dataclasses for TrainConfig / AugConfig + YAML loader
  train.py         single training entrypoint, config-driven
  test.py          benchmark a trained run on labels/test
  infer.py         video crops → Excel report
  models/
    base.py        ABC every model implements
    yolo_cls.py    yolo11{n,s}-cls fine-tuning backend
configs/<name>.yaml   one file per parameter set
results/<name>_<ts>/  one folder per run (weights, config copy, test report)
                      completed runs are moved to archive/classifier_experiments/
```

## Lifecycle of a run

```bash
# 1. Train. Drops everything into results/<name>_<timestamp>/.
python -m classifier.train --config configs/yolo_n_default.yaml

# 2. Test on the held-out split. Writes test_report.json + optional errors/.
python -m classifier.test --run results/roll_cls_angular_<ts> --save-errors

# 3. Score a video. Writes <crops_dir>.xlsx.
python -m classifier.infer \
    --run results/roll_cls_angular_<ts> \
    --crops-dir "crops/RGoldies_23_10_25"

```

## Adding a new config

Copy an existing YAML in `configs/`, rename, change knobs, run. The `name`
field becomes the run folder prefix — make it descriptive.

## Adding a new model type

1. Subclass `classifier.models.base.Classifier`.
2. Register the class in `classifier.models.__init__.MODEL_REGISTRY` under a
   string key.
3. Reference that key in a config's `model_type` field.

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
python -m classifier.train --config configs/roll_cls_angular.yaml

# 2. Test on the held-out split. Writes a report + optional errors/.
#    Pick the harness that matches the config's label space:
#      roll_cls           -> classifier.test_roll  (4-way roll)
#      multihead / angle_reg -> classifier.test    (16-way composite)
python -m classifier.test_roll --run results/roll_cls_angular_<ts>

# 3. Score a video. Writes <crops_dir>.xlsx.
python -m classifier.infer \
    --run results/roll_cls_angular_<ts> \
    --crops-dir "crops/RGoldies 18_9_25"

```

## Beyond roll

The roll classifier is the deliverable and the main
[README](../README.md) covers it. The other label spaces live here:

| Config | `model_type` | Test with |
|---|---|---|
| `roll_cls_angular.yaml`, `roll_cls.yaml` | `roll_cls` | `classifier.test_roll` |
| `multihead_decoupled.yaml`, `multihead_angular.yaml`, `multihead_default.yaml`, `multihead_reg.yaml`, `ablation_aug_{on,off}.yaml` | `multihead` | `classifier.test` |
| `angle_reg.yaml` | `angle_reg` | `classifier.test` (also reports heading error) |
| `yolo_n_default.yaml` | `yolo_cls` | `classifier.test` |

`angle_reg` additionally needs continuous heading annotations — see the heading
annotator in [webapp/README.md](../webapp/README.md).

**Active-learning miner.** `classifier.mine_candidates` ranks unlabeled crops by
how likely they are to be a rare class, and writes a queue the labeler serves
first via `--seed-queue`. It reads backbone embeddings and both head
distributions, so it works **only on a `multihead` run**:

```bash
python -m classifier.mine_candidates --run archive/classifier_experiments/multihead_decoupled_20260620_175813 \
    --top 200 --out seed_queue.txt
```

## Adding a new config

Copy an existing YAML in `configs/`, rename, change knobs, run. The `name`
field becomes the run folder prefix — make it descriptive.

## Adding a new model type

1. Subclass `classifier.models.base.Classifier`.
2. Register the class in `classifier.models.__init__.MODEL_REGISTRY` under a
   string key.
3. Reference that key in a config's `model_type` field.

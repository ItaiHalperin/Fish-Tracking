"""
Config dataclasses + YAML loader for the classifier pipeline.

Configs live under configs/<name>.yaml. The `name` field of the loaded config
seeds the results folder. To add a new parameter set, copy an existing YAML,
rename it, edit, run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class AugConfig:
    # Small continuous rotation — must not cross a class boundary.
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    # Direction-aware classes: keep both at 0.
    fliplr: float = 0.0
    flipud: float = 0.0
    mosaic: float = 0.0
    erasing: float = 0.0
    # Beta(alpha, alpha) mixup strength (multihead only). 0 = off.
    mixup: float = 0.0
    # Phase 2: rotate by random multiple of 45/90° and remap the class label.
    # Implemented in classifier/augmentation.py; off by default.
    cardinal_rotation: bool = False


@dataclass
class TrainConfig:
    name: str                      # results folder prefix
    model_type: str                # key in classifier/models/__init__.py
    pretrained: str = "yolo11n-cls.pt"
    data_dir: str = "labels"       # contains train/ val/ test/
    epochs: int = 50
    imgsz: int = 224
    batch: int = 32
    lr0: float = 0.01
    patience: int = 20
    seed: int = 42
    device: str = "mps"
    aug: AugConfig = field(default_factory=AugConfig)
    notes: str = ""                # human note that gets copied into results

    # --- factorized multi-head model (model_type: "multihead") ---
    backbone: str = "resnet34"     # any supported torchvision backbone
    weight_decay: float = 1e-4
    num_workers: int = 4
    # Imbalance handling: loss in {ce, class_balanced, focal}.
    loss: str = "class_balanced"
    cb_beta: float = 0.999         # effective-number reweighting strength
    focal_gamma: float = 2.0
    weighted_sampler: bool = True  # inverse-frequency oversampling of the tail
    # Regularization (overfit control)
    dropout: float = 0.0           # before the heads
    label_smoothing: float = 0.0   # CE only
    freeze_epochs: int = 0         # train heads only for the first N epochs
    # Decoupled two-stage (cRT): if > 0, after the main run, freeze the backbone,
    # reinit the heads, and retrain them for this many epochs on balanced sampling.
    # In two-stage mode stage 1 forces natural sampling + plain CE (best for
    # representations); stage 2 forces weighted sampling + class-balanced loss.
    decouple_epochs: int = 0
    # Inference-time (read at load(); editable post-hoc without retraining)
    tta: bool = False              # average over flip(remap) + small rotations
    abstain_threshold: float = 0.0  # below this conf, predict "unclear"

    # --- angle-regression model (model_type: "angle_reg") ---
    angle_map: str = "configs/angle_map.yaml"  # class -> (heading, roll) degrees
    headings_file: str = ""  # optional {crop: heading_deg} from the line tool


def load_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text())
    aug_data = data.pop("aug", {}) or {}
    return TrainConfig(aug=AugConfig(**aug_data), **data)


def dump_config(cfg: TrainConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

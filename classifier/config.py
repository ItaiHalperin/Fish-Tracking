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


def load_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text())
    aug_data = data.pop("aug", {}) or {}
    return TrainConfig(aug=AugConfig(**aug_data), **data)


def dump_config(cfg: TrainConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

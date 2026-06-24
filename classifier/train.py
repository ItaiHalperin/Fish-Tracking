"""
Single training entrypoint, config-driven.

Usage:
    python -m classifier.train --config configs/yolo_n_default.yaml
    python -m classifier.train --config configs/roll_cls.yaml --register

Drops everything for this run under results/<name>_<timestamp>/. With --register,
the run is written into the MLStorage classifier registry instead, so the
analysis pipeline can resolve it via storage.classifier_models.get_weights().
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import TrainConfig, load_config, dump_config
from .models import build_model


def run(config: TrainConfig, results_root: Path = Path("results"),
        output_dir: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = results_root / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the config we actually used into the results folder for reproducibility.
    dump_config(config, output_dir / "config.yaml")

    print(f"Training {config.name} → {output_dir}")
    model = build_model(config.model_type)
    weights_path = model.train(config, output_dir)
    print(f"Weights saved to {weights_path}")

    # Drop a stub metadata file; tests fill in test_report.json later.
    (output_dir / "run.json").write_text(json.dumps({
        "name": config.name,
        "model_type": config.model_type,
        "timestamp": timestamp,
        "weights": str(weights_path.relative_to(output_dir)),
    }, indent=2))

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--register", action="store_true",
                        help="Save the run into the MLStorage classifier registry")
    parser.add_argument("--model-name", default="fish_position_classifier",
                        help="Registry model name when --register is set")
    parser.add_argument("--storage-root", default="ml_storage")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.register:
        from core.ml_storage import MLStorage
        storage = MLStorage(args.storage_root)
        project, run_name = storage.classifier_models.prepare_run(args.model_name)
        out = run(cfg, output_dir=Path(project) / run_name)
        storage.classifier_models.finalize_run(args.model_name, out, metadata={
            "name": cfg.name, "model_type": cfg.model_type,
        })
    else:
        out = run(cfg)
    print(f"\nDone. Run dir: {out}")


if __name__ == "__main__":
    main()

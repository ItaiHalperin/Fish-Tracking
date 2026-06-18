"""
Single training entrypoint, config-driven.

Usage:
    python -m classifier.train --config configs/yolo_n_default.yaml

Drops everything for this run under results/<name>_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import TrainConfig, load_config, dump_config
from .models import build_model


def run(config: TrainConfig, results_root: Path = Path("results")) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run(cfg)
    print(f"\nDone. Run dir: {out}")


if __name__ == "__main__":
    main()

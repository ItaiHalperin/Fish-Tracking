"""
Benchmark a trained run on its test set.

Usage:
    python -m classifier.test --run results/yolo_n_default_20260618_120000
    python -m classifier.test --run results/<name> --save-errors

Reads <run>/config.yaml to know which model_type and data_dir to use, runs the
test split through it, writes <run>/test_report.json and optionally
<run>/errors/<true>__as__<pred>/<img>.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from . import angle_eval as AE
from . import angles as A
from . import data as D
from .config import load_config
from .models import build_model

DEFAULT_HEADINGS = Path("labels_angles/headings.json")


def collect_test_set(test_dir: Path, class_names: list[str]) -> tuple[list[Path], list[str]]:
    paths, truths = [], []
    skipped: list[str] = []
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if cls not in class_names:
            skipped.append(cls)
            continue
        for img in sorted(cls_dir.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(img)
                truths.append(cls)
    if skipped:
        print(f"  skipping test classes not in model: {skipped}")
    return paths, truths


def metrics(truths: list[str], preds: list[str], classes: list[str]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(truths, preds):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = {}
    macro_f1 = 0.0
    seen = 0
    for cls in sorted(set(classes) | set(truths)):
        support = sum(1 for t in truths if t == cls)
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": support}
        if support:
            macro_f1 += f1
            seen += 1

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(truths, preds):
        confusion[t][p] += 1

    return {
        "n_total": len(truths),
        "n_correct": sum(t == p for t, p in zip(truths, preds)),
        "accuracy": sum(t == p for t, p in zip(truths, preds)) / max(1, len(truths)),
        "macro_f1": macro_f1 / max(1, seen),
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def resolve_headings(cfg, override: str | None) -> dict[str, float]:
    """Continuous heading labels, for the angular report only. Classifier configs
    leave headings_file empty, so fall back to the standard location."""
    path = Path(override or cfg.headings_file or DEFAULT_HEADINGS)
    return D.load_headings(path) if path.is_file() else {}


def angle_report(model, paths: list[Path], truths: list[str], preds: list[str],
                 cfg, headings_override: str | None) -> dict | None:
    """Score the run in degrees. Skipped only if the class->angle map is missing."""
    if not Path(cfg.angle_map).is_file():
        print(f"  no angle map at {cfg.angle_map} — skipping angular metrics")
        return None
    angle_map = A.load_angle_map(cfg.angle_map)
    predict_angles = getattr(model, "predict_angles", None)
    needed = set(truths) if predict_angles else set(truths) | set(preds)
    if unmapped := needed - set(angle_map):
        print(f"  classes missing from {cfg.angle_map}: {sorted(unmapped)} — skipping angular metrics")
        return None

    pred_angles = predict_angles(paths) if predict_angles else None
    report = AE.angle_metrics([p.name for p in paths], truths,
                              None if pred_angles else preds, angle_map,
                              resolve_headings(cfg, headings_override),
                              pred_angles=pred_angles)
    n_invalid = sum(1 for p in preds if p not in angle_map and p != "unclear")
    report["invalid_composite_preds"] = n_invalid
    print(AE.format_report(report, "continuous" if pred_angles else "class centres"))
    if n_invalid:
        print(f"  note: {n_invalid}/{len(preds)} raw predictions were pose×facing "
              f"combinations absent from the label space (scored via the best valid class)")
    return report


def run(run_dir: Path, save_errors: bool = False,
        headings: str | None = None) -> dict:
    cfg = load_config(run_dir / "config.yaml")
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}")

    test_dir = Path(cfg.data_dir) / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"No test split at {test_dir}")

    model = build_model(cfg.model_type)
    model.load(weights)

    paths, truths = collect_test_set(test_dir, model.class_names)
    if not paths:
        raise SystemExit(f"No usable test images under {test_dir}")
    print(f"Testing {len(paths)} crops across {len(set(truths))} classes")

    predictions = model.predict(paths)
    preds = [p[0] for p in predictions]

    report = metrics(truths, preds, model.class_names)
    print(f"Accuracy: {report['accuracy']:.3%}    Macro F1: {report['macro_f1']:.3f}")
    report["angle_metrics"] = angle_report(model, paths, truths, preds, cfg, headings)
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=2))

    if save_errors:
        err_dir = run_dir / "errors"
        if err_dir.exists():
            shutil.rmtree(err_dir)
        n = 0
        for path, t, p in zip(paths, truths, preds):
            if t == p:
                continue
            bucket = err_dir / f"{t}__as__{p}"
            bucket.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bucket / path.name)
            n += 1
        print(f"Copied {n} misclassified crops into {err_dir}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to a results/<name> folder")
    parser.add_argument("--save-errors", action="store_true")
    parser.add_argument("--headings", default=None,
                        help=f"Continuous heading labels for the angular report "
                             f"(default: {DEFAULT_HEADINGS})")
    args = parser.parse_args()
    run(Path(args.run), save_errors=args.save_errors, headings=args.headings)


if __name__ == "__main__":
    main()

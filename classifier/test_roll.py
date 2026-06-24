"""
Evaluate a roll-only run on its test split.

The generic test.py compares predictions to composite class folders; roll has its
own label space, so this maps each test crop's class -> true roll and scores the
roll prediction. Reports overall roll accuracy, macro F1, the binary
"upside-down" accuracy, and a confusion matrix.

Usage:
    python -m classifier.test_roll --run results/roll_cls_<ts>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config
from .models import build_model
from . import data as D


def run(run_dir: Path) -> dict:
    cfg = load_config(run_dir / "config.yaml")
    model = build_model(cfg.model_type)
    model.load(run_dir / "weights" / "best.pt")

    test_s = D.load_split_from_dir(cfg.data_dir, "test")
    if not test_s:
        raise SystemExit(f"No test split at {cfg.data_dir}/test")

    # Models that predict more than roll (e.g. angle_reg) expose predict_roll so
    # we score only their roll head, comparable to the roll-only classifier.
    predict_roll = getattr(model, "predict_roll", model.predict)
    preds = predict_roll([s.path for s in test_s])
    truths = [model.true_roll_name(s.composite) for s in test_s]
    pred_names = [p[0] for p in preds]

    classes = sorted(set(truths) | set(pred_names))
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(truths, pred_names):
        confusion[t][p] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1; fn[t] += 1

    per_class, f1s = {}, []
    for c in classes:
        support = sum(1 for t in truths if t == c)
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)

    # binary upside-down (belly_up vs not)
    up_t = [t == "belly_up" for t in truths]
    up_p = [p == "belly_up" for p in pred_names]
    up_correct = sum(int(a == b) for a, b in zip(up_t, up_p))

    report = {
        "n_total": len(truths),
        "accuracy": sum(int(t == p) for t, p in zip(truths, pred_names)) / len(truths),
        "macro_f1": sum(f1s) / max(1, len(f1s)),
        "upside_down_accuracy": up_correct / len(truths),
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }
    (run_dir / "roll_test_report.json").write_text(json.dumps(report, indent=2))

    print(f"Roll test: {report['n_total']} crops")
    print(f"  roll accuracy:      {report['accuracy']*100:.2f}%")
    print(f"  roll macro F1:      {report['macro_f1']:.3f}")
    print(f"  upside-down acc:    {report['upside_down_accuracy']*100:.2f}%")
    print("  per class (f1 / recall / support):")
    for c in sorted(per_class, key=lambda x: per_class[x]["f1"]):
        v = per_class[c]
        if v["support"]:
            print(f"    {v['f1']:.2f}  r {v['recall']:.2f}  n {v['support']:>3}  {c}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run(Path(args.run))


if __name__ == "__main__":
    main()

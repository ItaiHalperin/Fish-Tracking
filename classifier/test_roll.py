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
from . import angle_eval as AE
from . import angles as A
from . import data as D


def roll_angle_metrics(model, samples, truths: list[str], pred_names: list[str],
                       cfg) -> dict:
    """How many degrees off the roll was, not just whether the class matched.
    A belly_down/flank confusion costs 90°; belly_down/belly_up costs 180°."""
    angle_map = A.load_angle_map(cfg.angle_map)
    true_degs = [angle_map[s.composite][1] for s in samples]
    predict_angles = getattr(model, "predict_angles", None)
    if predict_angles:
        pred_degs = [r for _, r in predict_angles([s.path for s in samples])]
    else:
        pred_degs = [AE.roll_deg_from_name(p) for p in pred_names]

    errors = [AE.circular_error(p, t) for p, t in zip(pred_degs, true_degs)]
    summary = AE.summarize(errors)
    summary["per_true_roll"] = {
        A.roll_name(deg): AE.summarize([e for e, t in zip(errors, true_degs) if t == deg])
        for deg in sorted(set(true_degs))
    }
    print(f"  roll angular err:   mean {summary['mean_err_deg']:.1f}°  "
          f"median {summary['median_err_deg']:.1f}°")
    for name, s in summary["per_true_roll"].items():
        if s["n"]:
            print(f"    {s['mean_err_deg']:6.1f}°  n {s['n']:>3}  {name}")
    return summary


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

    print(f"Roll test: {report['n_total']} crops")
    print(f"  roll accuracy:      {report['accuracy']*100:.2f}%")
    print(f"  roll macro F1:      {report['macro_f1']:.3f}")
    print(f"  upside-down acc:    {report['upside_down_accuracy']*100:.2f}%")
    print("  per class (f1 / recall / support):")
    for c in sorted(per_class, key=lambda x: per_class[x]["f1"]):
        v = per_class[c]
        if v["support"]:
            print(f"    {v['f1']:.2f}  r {v['recall']:.2f}  n {v['support']:>3}  {c}")

    report["roll_angle"] = roll_angle_metrics(model, test_s, truths, pred_names, cfg)
    (run_dir / "roll_test_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run(Path(args.run))


if __name__ == "__main__":
    main()

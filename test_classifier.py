#!/usr/bin/env python3
"""
Benchmark a trained classifier against a labeled test set.

Expects an ImageFolder-style layout:
    <test-dir>/
        <class_a>/*.jpg
        <class_b>/*.jpg
        ...

Reports:
  - overall accuracy
  - per-class precision / recall / F1 / support
  - confusion matrix (printed + saved as PNG)
  - copies misclassified crops into <output>/errors/<true>__as__<pred>/
    so you can eyeball where the model is failing
"""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


def load_test_set(test_dir: Path, model_classes: list[str]) -> list[tuple[Path, str]]:
    items = []
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if cls not in model_classes:
            print(f"  WARN: test class '{cls}' is not in model classes {model_classes} — skipped")
            continue
        for img in sorted(cls_dir.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((img, cls))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="best.pt from train_classifier.py")
    parser.add_argument("--test-dir", default="labels/test",
                        help="Folder of <class>/*.jpg (default: labels/test)")
    parser.add_argument("--output", default="runs/classify/test_report",
                        help="Where to write the confusion matrix and error crops")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save-errors", action="store_true",
                        help="Copy misclassified crops into <output>/errors/")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"Test dir not found: {test_dir}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names = list(model.names.values())

    items = load_test_set(test_dir, class_names)
    if not items:
        test_dirs = [p.name for p in test_dir.iterdir() if p.is_dir()]
        msg = f"No usable test images under {test_dir}."
        if test_dirs:
            msg += (f"\nTest dir contains classes {test_dirs} but the model"
                    f" was trained on {class_names}.\nRetrain with"
                    f" `python train_classifier.py --data labels` and try again.")
        raise SystemExit(msg)
    print(f"Loaded {len(items)} test crops across {len(set(c for _, c in items))} classes")

    # Predict in batches
    paths = [str(p) for p, _ in items]
    truths = [c for _, c in items]
    preds: list[str] = []
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        results = model.predict(source=chunk, device=args.device, verbose=False)
        for r in results:
            preds.append(model.names[int(r.probs.top1)])

    # Overall accuracy
    correct = sum(t == p for t, p in zip(truths, preds))
    acc = correct / len(items)
    print(f"\nOverall accuracy: {correct}/{len(items)} = {acc:.3%}")

    # Per-class precision/recall/F1
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()
    for t, p in zip(truths, preds):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    print(f"\n{'class':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    print("-" * 64)
    macro_f1 = 0.0
    n_classes_seen = 0
    for cls in sorted(set(truths) | set(preds)):
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if support[cls]:
            macro_f1 += f1
            n_classes_seen += 1
        print(f"{cls:<20} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {support[cls]:>10}")
    print("-" * 64)
    print(f"{'macro F1':<20} {'':>10} {'':>10} {macro_f1 / max(1, n_classes_seen):>10.3f}")

    # Confusion matrix (printed)
    print("\nConfusion matrix (rows = truth, cols = pred):")
    all_classes = sorted(set(class_names) | set(truths))
    matrix: dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(truths, preds):
        matrix[t][p] += 1
    header = " " * 20 + " ".join(f"{c[:8]:>9}" for c in all_classes)
    print(header)
    for t in all_classes:
        row = f"{t:<20}" + " ".join(f"{matrix[t][p]:>9d}" for p in all_classes)
        print(row)

    # Confusion matrix PNG (optional — only if matplotlib is available)
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        n = len(all_classes)
        cm = np.zeros((n, n), dtype=int)
        for i, t in enumerate(all_classes):
            for j, p in enumerate(all_classes):
                cm[i, j] = matrix[t][p]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_xticklabels(all_classes, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(all_classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im)
        plt.tight_layout()
        png_path = output / "confusion_matrix.png"
        plt.savefig(png_path, dpi=120)
        print(f"\nConfusion matrix PNG: {png_path}")
    except ImportError:
        print("\n(install matplotlib to also get a confusion_matrix.png)")

    # Misclassified crops
    if args.save_errors:
        err_dir = output / "errors"
        if err_dir.exists():
            shutil.rmtree(err_dir)
        n_err = 0
        for (path, t), p in zip(items, preds):
            if t == p:
                continue
            bucket = err_dir / f"{t}__as__{p}"
            bucket.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bucket / path.name)
            n_err += 1
        print(f"\nCopied {n_err} misclassified crops into {err_dir}")


if __name__ == "__main__":
    main()

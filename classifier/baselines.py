"""
Baselines: establish the floor the real models must beat.

Two baselines x two tasks (16-way composite full-position, 4-way roll), on the
same labels_split as the trained models:
  1. Majority class  — always predict the most frequent train class.
  2. Logistic regression on raw pixels — flatten + standardize 32x32 RGB crops,
     fit multinomial logistic regression (a single linear+softmax layer, the
     torch equivalent of sklearn's LogisticRegression; no sklearn dependency).

A baseline that scores well means the task is easy; the gap to our models shows
how much the learned features actually buy.

Usage:
    python -m classifier.baselines --data-dir labels_split
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from . import data as D
from . import angles as A


def _features(samples, imgsz: int) -> torch.Tensor:
    feats = []
    for s in samples:
        img = Image.open(s.path).convert("RGB").resize((imgsz, imgsz))
        feats.append(torch.frombuffer(img.tobytes(), dtype=torch.uint8).float() / 255.0)
    return torch.stack(feats)


def _macro_f1(truth: list[int], pred: list[int], k: int) -> float:
    tp = Counter(); fp = Counter(); fn = Counter()
    for t, p in zip(truth, pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1; fn[t] += 1
    f1s = []
    for c in range(k):
        if sum(1 for t in truth if t == c) == 0:
            continue
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / max(1, len(f1s))


def _acc(truth: list[int], pred: list[int]) -> float:
    return sum(int(t == p) for t, p in zip(truth, pred)) / max(1, len(truth))


def _logreg(Xtr, ytr, Xte, k, seed, epochs=300):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    clf = torch.nn.Linear(Xtr.shape[1], k)
    opt = torch.optim.AdamW(clf.parameters(), lr=0.01, weight_decay=1e-3)
    yt = torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(Xtr), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return clf(Xte).argmax(1).tolist()


def _eval_task(name, ytr, yte, k, Xtr, Xte, seed) -> dict:
    maj = Counter(ytr).most_common(1)[0][0]
    maj_pred = [maj] * len(yte)
    lr_pred = _logreg(Xtr, ytr, Xte, k, seed)
    out = {
        "task": name, "n_test": len(yte), "n_classes": k,
        "majority": {"acc": _acc(yte, maj_pred), "macro_f1": _macro_f1(yte, maj_pred, k)},
        "logreg_pixels": {"acc": _acc(yte, lr_pred), "macro_f1": _macro_f1(yte, lr_pred, k)},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="labels_split")
    ap.add_argument("--angle-map", default="configs/angle_map.yaml")
    ap.add_argument("--imgsz", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()

    train_s = D.load_split_from_dir(args.data_dir, "train")
    test_s = D.load_split_from_dir(args.data_dir, "test")
    if not train_s or not test_s:
        raise SystemExit(f"Need train and test splits under {args.data_dir}")
    print(f"train {len(train_s)}  test {len(test_s)}  (imgsz {args.imgsz})")

    Xtr, Xte = _features(train_s, args.imgsz), _features(test_s, args.imgsz)

    comp_vocab = sorted({s.composite for s in train_s + test_s})
    ci = {c: i for i, c in enumerate(comp_vocab)}
    comp = _eval_task("composite (16-way)",
                      [ci[s.composite] for s in train_s], [ci[s.composite] for s in test_s],
                      len(comp_vocab), Xtr, Xte, args.seed)

    amap = A.load_angle_map(args.angle_map)
    rolls = A.roll_vocab(amap)
    ri = {r: i for i, r in enumerate(rolls)}
    roll = _eval_task("roll (4-way)",
                      [ri[amap[s.composite][1]] for s in train_s],
                      [ri[amap[s.composite][1]] for s in test_s],
                      len(rolls), Xtr, Xte, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"composite": comp, "roll": roll}, indent=2))

    print(f"\n{'task':22} {'baseline':16} {'acc':>7} {'macroF1':>8}")
    print("-" * 56)
    for r in (comp, roll):
        for b in ("majority", "logreg_pixels"):
            print(f"{r['task']:22} {b:16} {r[b]['acc']*100:6.1f}% {r[b]['macro_f1']:8.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

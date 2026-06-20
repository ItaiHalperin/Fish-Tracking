"""
Mine likely tail-class crops from the unlabeled pool to seed the labeler.

Combines three signals (active learning for rare classes):
  1. kNN retrieval  — crops nearest (cosine) to labeled tail-class prototypes in
     backbone feature space. Best yield: works even when the classifier head is
     weak on the tail.
  2. tail prediction — crops the classifier predicts as a tail class.
  3. uncertainty     — crops where the heads are unsure (high pose/facing entropy).

Writes a prioritized `seed_queue.txt` (flat basenames the webapp already consumes)
plus a `mine_report.json` recording why each crop was picked. A human still
verifies every crop in the labeler — these are candidates, not labels.

Usage:
    python -m classifier.mine_candidates --run results/<dir> \
        --pool crops/all_flat labels_raw/unclear --top 200
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .data import _IMG_EXT, _track_of
from .models import build_model
from .models.multihead import _resolve_device
from .config import load_config
from . import labels as L


def _list_pool(pool_dirs: list[str], limit: int | None, per_track: int,
               seed: int) -> list[Path]:
    paths: list[Path] = []
    for d in pool_dirs:
        root = Path(d)
        if not root.is_dir():
            print(f"  WARNING: pool dir not found: {root}")
            continue
        found = [p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT]
        print(f"  {root}: {len(found)} crops")
        paths.extend(found)

    # Cap per fish track so the queue isn't 50 frames of one fish.
    by_track: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in paths:
        by_track[_track_of(p)].append(p)
    rng = random.Random(seed)
    capped: list[Path] = []
    for group in by_track.values():
        rng.shuffle(group)
        capped.extend(group[:per_track])

    rng.shuffle(capped)
    if limit and len(capped) > limit:
        capped = capped[:limit]
    print(f"  pool after per-track cap ({per_track}) + limit: {len(capped)} crops")
    return capped


def _tail_classes(raw_dir: Path, threshold: int) -> list[str]:
    counts = {}
    for d in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if d.name == "unclear":
            continue
        counts[d.name] = sum(1 for f in d.iterdir() if f.suffix.lower() in _IMG_EXT)
    return [c for c, n in counts.items() if n < threshold]


def _prototypes(model, raw_dir: Path, tail: list[str], max_per_class: int):
    """Mean L2-normalized embedding per tail class."""
    protos, names = [], []
    for cls in tail:
        files = [f for f in (raw_dir / cls).iterdir() if f.suffix.lower() in _IMG_EXT]
        if not files:
            continue
        files = files[:max_per_class]
        _, emb, _, _ = model.embed_and_predict(files)
        if len(emb) == 0:
            continue
        protos.append(torch.nn.functional.normalize(emb.mean(0), dim=0))
        names.append(cls)
    return names, (torch.stack(protos) if protos else torch.empty(0))


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-9).log() * probs).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Trained run dir (uses its weights)")
    ap.add_argument("--pool", nargs="+", default=["crops/all_flat", "labels_raw/unclear"])
    ap.add_argument("--raw-dir", default="labels_raw")
    ap.add_argument("--out", default="seed_queue.txt")
    ap.add_argument("--top", type=int, default=200, help="Total crops to queue")
    ap.add_argument("--limit", type=int, default=40000, help="Max pool crops to scan")
    ap.add_argument("--per-track", type=int, default=3, help="Max candidates per fish track")
    ap.add_argument("--tail-threshold", type=int, default=90,
                    help="Classes with fewer labeled crops than this are 'tail'")
    ap.add_argument("--proto-cap", type=int, default=60, help="Max labeled crops per prototype")
    ap.add_argument("--device", default=None, help="Override device (default: run config's)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    model = build_model(cfg.model_type)
    model.load(run_dir / "weights" / "best.pt")
    # Inference defaults to CPU; move to the run's device so scanning a big pool
    # isn't glacial.
    dev = _resolve_device(args.device or cfg.device)
    model._device = dev
    model._net.to(dev)
    print(f"Mining on device: {dev}")

    raw = Path(args.raw_dir)
    tail = _tail_classes(raw, args.tail_threshold)
    print(f"Tail classes (<{args.tail_threshold} labeled): {tail}")
    tail_set = set(tail)

    print("Building tail-class prototypes...")
    proto_names, protos = _prototypes(model, raw, tail, args.proto_cap)

    print("Listing pool...")
    pool = _list_pool(args.pool, args.limit, args.per_track, args.seed)
    if not pool:
        raise SystemExit("Empty pool — nothing to mine.")

    print(f"Embedding {len(pool)} crops...")
    kept, emb, pose_p, facing_p = model.embed_and_predict(pool)
    print(f"  embedded {len(kept)} crops")

    # Signal 1: kNN similarity to nearest tail prototype.
    if len(protos):
        knn = (emb @ protos.T).max(1).values  # [N]
        knn_proto = proto_names and [proto_names[i] for i in (emb @ protos.T).argmax(1)]
    else:
        knn = torch.zeros(len(kept))
        knn_proto = ["" for _ in kept]

    # Signal 2: predicted composite is a tail class.
    pose_idx = pose_p.argmax(1)
    facing_idx = facing_p.argmax(1)
    poses, facings = model._vocab.poses, model._vocab.facings
    pred = [L.compose(poses[pi], facings[fi]) for pi, fi in zip(pose_idx, facing_idx)]
    pred_conf = (pose_p.max(1).values * facing_p.max(1).values)

    # Signal 3: head uncertainty (combined normalized entropy).
    unc = _entropy(pose_p) / max(1e-9, float(_entropy(pose_p).max())) \
        + _entropy(facing_p) / max(1e-9, float(_entropy(facing_p).max()))

    # Rank within each signal, then take a balanced quota from each (dedup).
    order_knn = sorted(range(len(kept)), key=lambda i: -float(knn[i]))
    order_pred = [i for i in sorted(range(len(kept)), key=lambda i: -float(pred_conf[i]))
                  if pred[i] in tail_set]
    order_unc = sorted(range(len(kept)), key=lambda i: -float(unc[i]))

    quota = {"knn": args.top // 2, "pred": args.top // 4, "unc": args.top - args.top // 2 - args.top // 4}
    picked: list[int] = []
    seen: set[int] = set()
    for name, order in [("knn", order_knn), ("pred", order_pred), ("unc", order_unc)]:
        added = 0
        for i in order:
            if i in seen:
                continue
            seen.add(i)
            picked.append(i)
            reason = name
            added += 1
            if added >= quota[name]:
                break

    report = []
    for i in picked:
        report.append({
            "file": kept[i].name,
            "knn_sim": round(float(knn[i]), 3),
            "knn_nearest_tail": knn_proto[i] if knn_proto else "",
            "predicted": pred[i],
            "pred_is_tail": pred[i] in tail_set,
            "pred_conf": round(float(pred_conf[i]), 3),
            "uncertainty": round(float(unc[i]), 3),
        })

    Path(args.out).write_text("\n".join(kept[i].name for i in picked) + "\n")
    Path(run_dir / "mine_report.json").write_text(json.dumps({
        "run": str(run_dir), "tail_classes": tail,
        "n_pool_embedded": len(kept), "n_queued": len(picked),
        "quota": quota, "candidates": report,
    }, indent=2))

    by_pred = Counter(pred[i] for i in picked)
    print(f"\nWrote {len(picked)} candidates to {args.out}")
    print(f"  report: {run_dir / 'mine_report.json'}")
    print("  predicted-class breakdown of the queue:")
    for c, n in by_pred.most_common():
        mark = " (tail)" if c in tail_set else ""
        print(f"    {n:>3}  {c}{mark}")


if __name__ == "__main__":
    main()

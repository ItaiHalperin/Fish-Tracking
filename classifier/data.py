"""
Build a leak-free train/val/test split from labels_raw/, and provide the torch
Dataset used by the factorized (pose + facing) classifier.

Crops are named `<video>__id_<N>_frame_<M>.jpg`. All crops of one fish (a
(video, id) track) are near-duplicates, so splits are made by *track*, never by
crop — splitting by crop puts the same fish in train and test and inflates
accuracy. The split is global across classes because one track spans several
composite classes as the fish changes pose.

CLI (materializes the split to disk so every model + test.py share it):
    python -m classifier.data --out labels --val-ratio 0.2 --test-ratio 0.15 \
        --skip-class unclear

Library: scan / split_by_track / FishCropDataset / weighted_sampler, plus
load_split_from_dir to read a materialized split back.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler

from . import labels as L

_TRACK_RE = re.compile(r"^(?P<video>.+)__id_(?P<id>\d+)_frame_\d+", re.IGNORECASE)
_IMG_EXT = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    path: Path
    composite: str
    pose: str
    facing: str
    track: tuple[str, str]


@dataclass
class Vocab:
    poses: list[str]
    facings: list[str]
    composites: list[str]

    def pose_idx(self, p: str) -> int:
        return self.poses.index(p)

    def facing_idx(self, f: str) -> int:
        return self.facings.index(f)


def _track_of(path: Path) -> tuple[str, str]:
    m = _TRACK_RE.match(path.stem)
    if not m:
        return (path.stem, "")
    return (m.group("video"), m.group("id"))


def _sample(path: Path, composite: str) -> Sample:
    pose, facing = L.parse_composite(composite)
    return Sample(path, composite, pose, facing, _track_of(path))


def scan(source_dir: str | Path, skip: list[str]) -> list[Sample]:
    source = Path(source_dir)
    skipped = set(skip)
    out: list[Sample] = []
    for cls_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        if cls_dir.name in skipped:
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in _IMG_EXT:
                out.append(_sample(img, cls_dir.name))
    return out


def load_split_from_dir(data_dir: str | Path, split: str) -> list[Sample]:
    """Read a materialized split (data_dir/<split>/<composite>/*.jpg)."""
    split_dir = Path(data_dir) / split
    out: list[Sample] = []
    if not split_dir.is_dir():
        return out
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in _IMG_EXT:
                out.append(_sample(img, cls_dir.name))
    return out


def build_vocab(samples: list[Sample]) -> Vocab:
    return Vocab(
        poses=sorted({s.pose for s in samples}),
        facings=sorted({s.facing for s in samples}),
        composites=sorted({s.composite for s in samples}),
    )


def split_by_track(samples: list[Sample], val_ratio: float, test_ratio: float,
                   seed: int) -> dict[str, list[Sample]]:
    """Assign whole tracks to splits (leak-free), class-aware so no class ends up
    absent from training. Classes are placed rarest-first and each is guaranteed
    at least one track in train; a track is fixed the first time any class claims
    it, so common classes fill the remaining ratio budget."""
    by_track: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    class_tracks: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for s in samples:
        by_track[s.track].append(s)
        class_tracks[s.composite].add(s.track)

    class_total = Counter(s.composite for s in samples)
    rng = random.Random(seed)
    assigned: dict[tuple[str, str], str] = {}

    for cls in sorted(class_total, key=lambda c: class_total[c]):
        free = [t for t in class_tracks[cls] if t not in assigned]
        rng.shuffle(free)
        if not free:
            continue
        assigned[free[0]] = "train"  # guarantee train coverage for this class
        rest = free[1:]
        n_test = int(round(len(rest) * test_ratio))
        n_val = int(round(len(rest) * val_ratio))
        for t in rest[:n_test]:
            assigned[t] = "test"
        for t in rest[n_test:n_test + n_val]:
            assigned[t] = "val"
        for t in rest[n_test + n_val:]:
            assigned[t] = "train"

    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for track, group in by_track.items():
        splits[assigned.get(track, "train")].extend(group)
    return splits


def split_report(splits: dict[str, list[Sample]]) -> str:
    classes = sorted({s.composite for g in splits.values() for s in g})
    counts = {name: Counter(s.composite for s in g) for name, g in splits.items()}
    width = max([len(c) for c in classes] + [len("class")])
    lines = [f"{'class':<{width}}  {'train':>6} {'val':>5} {'test':>5}"]
    lines.append("-" * len(lines[0]))
    for c in classes:
        lines.append(f"{c:<{width}}  {counts['train'][c]:>6} "
                     f"{counts['val'][c]:>5} {counts['test'][c]:>5}")
    lines.append("-" * len(lines[1]))
    lines.append(f"{'TOTAL':<{width}}  {len(splits['train']):>6} "
                 f"{len(splits['val']):>5} {len(splits['test']):>5}")
    return "\n".join(lines)


class FishCropDataset(Dataset):
    """
    Returns (image_tensor, pose_idx, facing_idx). On train, a random horizontal
    flip is applied with its labels remapped (left<->right) — a genuine extra
    view, not a relabeling error, which is why flips are usable here but not in
    the single-head pipeline.
    """

    def __init__(self, samples: list[Sample], vocab: Vocab, transform,
                 flip_p: float = 0.0, rng_seed: int = 0):
        self.samples = samples
        self.vocab = vocab
        self.transform = transform
        self.flip_p = flip_p
        self._rng = random.Random(rng_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        pose, facing = s.pose, s.facing
        if self.flip_p and self._rng.random() < self.flip_p:
            img = ImageOps.mirror(img)
            pose = L.flip_pose_lr(pose)
            facing = L.flip_facing_lr(facing)
        return (self.transform(img),
                self.vocab.pose_idx(pose),
                self.vocab.facing_idx(facing))


def load_headings(path: str | Path) -> dict[str, float]:
    """{crop_basename: heading_deg} from the line-annotation tool. Missing file
    is fine — the dataset falls back to class-center headings."""
    p = Path(path)
    if not p or not p.is_file():
        return {}
    import json
    return {k: float(v) for k, v in json.loads(p.read_text()).items()}


class FishAngleDataset(Dataset):
    """Returns (image_tensor, heading_deg, roll_deg) for the regression model.

    heading comes from the per-crop line annotation when available, else the
    class-center value; roll always comes from the class (no roll relabeling).
    A horizontal flip reflects heading (h -> 180-h) and reverses roll
    (r -> 360-r: right flank <-> left flank, belly up/down unchanged)."""

    def __init__(self, samples: list[Sample], angle_map: dict[str, tuple[float, float]],
                 transform, flip_p: float = 0.0, rng_seed: int = 0,
                 headings: dict[str, float] | None = None):
        self.samples = samples
        self.angle_map = angle_map
        self.transform = transform
        self.flip_p = flip_p
        self._rng = random.Random(rng_seed)
        self.headings = headings or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        heading_center, roll = self.angle_map[s.composite]
        heading = self.headings.get(s.path.name, heading_center)
        if self.flip_p and self._rng.random() < self.flip_p:
            img = ImageOps.mirror(img)
            heading = (180.0 - heading) % 360.0
            roll = (360.0 - roll) % 360.0
        return self.transform(img), float(heading), float(roll)


class FishRollDataset(Dataset):
    """Returns (image_tensor, roll_class_idx) for the roll-only model. Roll comes
    from the class via angle_map (no roll relabeling). A horizontal flip swaps
    left/right flank (roll -> 360-r); belly up/down are unchanged."""

    def __init__(self, samples: list[Sample], angle_map: dict[str, tuple[float, float]],
                 roll_classes: list[float], transform, flip_p: float = 0.0, rng_seed: int = 0):
        self.samples = samples
        self.angle_map = angle_map
        self.roll_classes = roll_classes
        self.transform = transform
        self.flip_p = flip_p
        self._rng = random.Random(rng_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def roll_of(self, s: Sample, flip: bool) -> float:
        roll = self.angle_map[s.composite][1]
        return (360.0 - roll) % 360.0 if flip else roll

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        flip = bool(self.flip_p) and self._rng.random() < self.flip_p
        if flip:
            img = ImageOps.mirror(img)
        roll = self.roll_of(s, flip)
        return self.transform(img), self.roll_classes.index(roll)


def weighted_sampler(samples: list[Sample], seed: int) -> WeightedRandomSampler:
    """Inverse-frequency sampling over composite class to surface the tail."""
    freq = Counter(s.composite for s in samples)
    weights = [1.0 / freq[s.composite] for s in samples]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(samples),
                                 replacement=True, generator=g)


def _materialize(splits: dict[str, list[Sample]], out: Path, train_cap: int | None,
                 seed: int) -> dict:
    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, group in splits.items():
        if split_name == "train" and train_cap is not None:
            by_cls: dict[str, list[Sample]] = defaultdict(list)
            for s in group:
                by_cls[s.composite].append(s)
            kept: list[Sample] = []
            for cls, items in by_cls.items():
                if len(items) > train_cap:
                    rng.shuffle(items)
                    items = items[:train_cap]
                kept.extend(items)
            group = kept
        for s in group:
            d = out / split_name / s.composite
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s.path, d / s.path.name)
            summary[s.composite][split_name] += 1
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="labels_raw")
    parser.add_argument("--out", required=True, help="Output split folder (e.g. labels)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--train-cap", type=int, default=None,
                        help="Optional max crops per class in the TRAIN split only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-class", action="append", default=[],
                        help="Class to exclude entirely (repeat to skip several)")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    if not raw.is_dir():
        raise SystemExit(f"Raw folder not found: {raw}")
    if out.exists():
        shutil.rmtree(out)

    samples = scan(raw, args.skip_class)
    if not samples:
        raise SystemExit(f"No crops found under {raw} (after skipping {args.skip_class})")
    splits = split_by_track(samples, args.val_ratio, args.test_ratio, args.seed)

    print(split_report(splits))
    summary = _materialize(splits, out, args.train_cap, args.seed)

    (out / "split_info.json").write_text(json.dumps({
        "raw": str(raw.resolve()),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_cap": args.train_cap,
        "seed": args.seed,
        "skip": args.skip_class,
        "split_by": "track",
        "per_class": summary,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

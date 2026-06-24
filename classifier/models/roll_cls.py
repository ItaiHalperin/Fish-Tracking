"""
Roll-only classifier.

Predicts just the roll state (belly_down / right_flank / belly_up / left_flank) —
rotation about the body's long axis — which is the project's actual goal. Roll
targets come from configs/angle_map.yaml via each crop's existing class, so no
new labeling is needed. Use this to test whether the heading axis is needed at
all: compare its roll accuracy to the joint model's.

model_type: "roll_cls". Evaluate with: python -m classifier.test_roll --run <dir>
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from .base import Classifier
from .multihead import _build_backbone, _resolve_device, _transforms
from .. import data as D
from .. import angles as A
from ..losses import build_loss, class_balanced_weights


class _RollNet(nn.Module):
    def __init__(self, backbone_name: str, n_roll: int, dropout: float = 0.0,
                 pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, n_roll)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


class RollClassifier(Classifier):
    def __init__(self):
        self._net: _RollNet | None = None
        self._roll_classes: list[float] = []
        self._angle_map: dict[str, tuple[float, float]] = {}
        self._device = torch.device("cpu")
        self._imgsz = 224
        self._eval_tf = None

    @property
    def class_names(self) -> list[str]:
        return [A.roll_name(r) for r in self._roll_classes]

    def train(self, config, output_dir: Path) -> Path:
        torch.manual_seed(config.seed)
        self._device = _resolve_device(config.device)
        self._imgsz = config.imgsz
        self._angle_map = A.load_angle_map(config.angle_map)
        self._roll_classes = A.roll_vocab(self._angle_map)

        train_s = D.load_split_from_dir(config.data_dir, "train")
        val_s = D.load_split_from_dir(config.data_dir, "val")
        if not train_s:
            raise SystemExit(f"No training crops under {config.data_dir}/train.")

        rc = self._roll_classes
        train_ds = D.FishRollDataset(train_s, self._angle_map, rc, _transforms(config, True),
                                     flip_p=config.aug.fliplr, rng_seed=config.seed)
        val_ds = D.FishRollDataset(val_s, self._angle_map, rc, _transforms(config, False))
        sampler = D.weighted_sampler(train_s, config.seed) if config.weighted_sampler else None
        train_loader = DataLoader(train_ds, batch_size=config.batch, sampler=sampler,
                                  shuffle=sampler is None, num_workers=config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=config.batch, shuffle=False,
                                num_workers=config.num_workers)

        self._net = _RollNet(config.backbone, len(rc), dropout=config.dropout).to(self._device)

        counts = Counter(self._angle_map[s.composite][1] for s in train_s)
        weight = None
        if config.loss != "ce":
            weight = class_balanced_weights([counts.get(r, 0) for r in rc],
                                            config.cb_beta).to(self._device)
        loss_fn = build_loss(config.loss, weight, config.focal_gamma,
                             config.label_smoothing).to(self._device)

        opt = torch.optim.AdamW(self._net.parameters(), lr=config.lr0,
                                weight_decay=config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)

        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "best.pt"

        best_f1, best_epoch, history = -1.0, -1, []
        for epoch in range(config.epochs):
            self._net.train()
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(self._device), y.to(self._device)
                opt.zero_grad()
                loss = loss_fn(self._net(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
            sched.step()

            m = self._evaluate(val_loader)
            history.append({"epoch": epoch, "train_loss": running / max(1, len(train_ds)), **m})
            flag = ""
            if m["macro_f1"] > best_f1:
                best_f1, best_epoch = m["macro_f1"], epoch
                self._save(weights_path, config)
                flag = "  *"
            print(f"  epoch {epoch:>3}  loss {running/max(1,len(train_ds)):.3f}  "
                  f"roll_acc {m['acc']:.3f}  macroF1 {m['macro_f1']:.3f}  "
                  f"upside_down_acc {m['upside_acc']:.3f}{flag}")
            if epoch - best_epoch >= config.patience:
                print(f"  early stop (no val gain for {config.patience} epochs)")
                break

        (output_dir / "roll_report.json").write_text(json.dumps({
            "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
            "roll_classes": [A.roll_name(r) for r in rc],
            "n_train": len(train_s), "n_val": len(val_s), "history": history,
        }, indent=2))
        if not weights_path.exists():
            raise RuntimeError("Training produced no checkpoint.")
        return weights_path

    @torch.no_grad()
    def _evaluate(self, loader) -> dict:
        self._net.eval()
        tp = Counter(); fp = Counter(); fn = Counter()
        up_ok = up_total = correct = total = 0
        up_idx = self._roll_classes.index(180.0) if 180.0 in self._roll_classes else -1
        for x, y in loader:
            pred = self._net(x.to(self._device)).argmax(1).cpu()
            for t, p in zip(y.tolist(), pred.tolist()):
                correct += int(t == p)
                total += 1
                if t == p:
                    tp[t] += 1
                else:
                    fp[p] += 1; fn[t] += 1
                if up_idx >= 0:
                    up_total += int(t == up_idx)
                    if t == up_idx:
                        up_ok += int(p == up_idx)
        f1s = []
        for c in range(len(self._roll_classes)):
            prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
            rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
            if tp[c] + fn[c] > 0:
                f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return {"acc": correct / max(1, total), "macro_f1": sum(f1s) / max(1, len(f1s)),
                "upside_acc": up_ok / max(1, up_total)}

    def _save(self, path: Path, config) -> None:
        torch.save({"backbone": config.backbone, "imgsz": config.imgsz,
                    "roll_classes": self._roll_classes, "angle_map": self._angle_map,
                    "state_dict": self._net.state_dict()}, path)

    def load(self, weights_path: Path) -> None:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        self._roll_classes = ckpt["roll_classes"]
        self._angle_map = ckpt["angle_map"]
        self._imgsz = ckpt["imgsz"]
        self._net = _RollNet(ckpt["backbone"], len(self._roll_classes), pretrained=False)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        from types import SimpleNamespace
        self._eval_tf = _transforms(SimpleNamespace(imgsz=self._imgsz), train=False)

    @torch.no_grad()
    def predict(self, image_paths: Iterable[Path | str]) -> list[tuple[str, float]]:
        if self._net is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        from ..imageio import to_pil
        out = []
        for p in image_paths:
            x = self._eval_tf(to_pil(p)).unsqueeze(0).to(self._device)
            probs = torch.softmax(self._net(x), 1)[0]
            i = int(probs.argmax())
            out.append((A.roll_name(self._roll_classes[i]), float(probs[i])))
        return out

    def true_roll_name(self, composite: str) -> str:
        return A.roll_name(self._angle_map[composite][1])

"""
Orientation as regression (Option 3): one backbone, two angle heads.

Instead of classifying a composite label, the model regresses two continuous
angles — heading (in-plane head direction) and roll (rotation about the body
axis) — each as a (cos, sin) vector trained with an angular loss. This removes
the starved-tail-class problem (rare orientations are just nearby angles, not
empty buckets) and handles borderline poses smoothly.

Ground truth comes from the class -> (heading, roll) table in configs/angle_map.yaml,
so existing labels are reused with no relabeling. predict() snaps the regressed
angles back to the nearest class so it slots into the same test harness.

model_type: "angle_reg".
"""

from __future__ import annotations

import json
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


class _TwoAngleNet(nn.Module):
    def __init__(self, backbone_name: str, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.heading_head = nn.Linear(feat_dim, 2)  # (cos, sin)
        self.roll_head = nn.Linear(feat_dim, 2)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.heading_head(f), self.roll_head(f)


class AngleRegClassifier(Classifier):
    def __init__(self):
        self._net: _TwoAngleNet | None = None
        self._angle_map: dict[str, tuple[float, float]] = {}
        self._device = torch.device("cpu")
        self._imgsz = 224
        self._eval_tf = None

    @property
    def class_names(self) -> list[str]:
        if not self._angle_map:
            raise RuntimeError("Model not trained or loaded.")
        return list(self._angle_map)

    def train(self, config, output_dir: Path) -> Path:
        torch.manual_seed(config.seed)
        self._device = _resolve_device(config.device)
        self._imgsz = config.imgsz
        self._angle_map = A.load_angle_map(config.angle_map)

        train_s = D.load_split_from_dir(config.data_dir, "train")
        val_s = D.load_split_from_dir(config.data_dir, "val")
        if not train_s:
            raise SystemExit(f"No training crops under {config.data_dir}/train.")
        missing = {s.composite for s in train_s + val_s} - set(self._angle_map)
        if missing:
            raise SystemExit(f"Classes missing from {config.angle_map}: {sorted(missing)}")

        headings = D.load_headings(config.headings_file) if config.headings_file else {}
        if headings:
            n = sum(1 for s in train_s + val_s if s.path.name in headings)
            print(f"  using {len(headings)} line-annotated headings "
                  f"({n}/{len(train_s) + len(val_s)} of train+val matched)")

        train_ds = D.FishAngleDataset(train_s, self._angle_map, _transforms(config, True),
                                      flip_p=config.aug.fliplr, rng_seed=config.seed,
                                      headings=headings)
        val_ds = D.FishAngleDataset(val_s, self._angle_map, _transforms(config, False),
                                    headings=headings)
        sampler = D.weighted_sampler(train_s, config.seed) if config.weighted_sampler else None
        train_loader = DataLoader(train_ds, batch_size=config.batch, sampler=sampler,
                                  shuffle=sampler is None, num_workers=config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=config.batch, shuffle=False,
                                num_workers=config.num_workers)

        self._net = _TwoAngleNet(config.backbone, dropout=config.dropout).to(self._device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=config.lr0,
                                weight_decay=config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)

        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "best.pt"

        best_err = 1e9
        best_epoch = -1
        history = []
        for epoch in range(config.epochs):
            self._net.train()
            running = 0.0
            for x, hd, rd in train_loader:
                x = x.to(self._device)
                hd, rd = hd.to(self._device).float(), rd.to(self._device).float()
                opt.zero_grad()
                hv, rv = self._net(x)
                loss = A.angular_loss(hv, hd) + A.angular_loss(rv, rd)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
            sched.step()

            metrics = self._evaluate(val_loader)
            history.append({"epoch": epoch, "train_loss": running / max(1, len(train_ds)),
                            **metrics})
            mean_err = metrics["heading_err_deg"] + metrics["roll_err_deg"]
            flag = ""
            if mean_err < best_err:
                best_err = mean_err
                best_epoch = epoch
                self._save(weights_path, config)
                flag = "  *"
            print(f"  epoch {epoch:>3}  loss {running/max(1,len(train_ds)):.3f}  "
                  f"heading_err {metrics['heading_err_deg']:.1f}°  "
                  f"roll_err {metrics['roll_err_deg']:.1f}°  "
                  f"snap_acc {metrics['snap_acc']:.3f}{flag}")
            if epoch - best_epoch >= config.patience:
                print(f"  early stop (no val gain for {config.patience} epochs)")
                break

        (output_dir / "angle_report.json").write_text(json.dumps({
            "best_epoch": best_epoch, "best_val_total_err_deg": best_err,
            "n_train": len(train_s), "n_val": len(val_s), "history": history,
        }, indent=2))
        if not weights_path.exists():
            raise RuntimeError("Training produced no checkpoint.")
        return weights_path

    @torch.no_grad()
    def _evaluate(self, loader) -> dict:
        self._net.eval()
        h_err = r_err = snap_ok = total = 0.0
        for x, hd, rd in loader:
            x = x.to(self._device)
            hv, rv = self._net(x)
            hp = A.vec_to_deg(hv).cpu()
            rp = A.vec_to_deg(rv).cpu()
            h_err += A.angular_error_deg(hp, hd).sum().item()
            r_err += A.angular_error_deg(rp, rd).sum().item()
            for i in range(x.size(0)):
                snap, _ = A.snap_to_class(float(hp[i]), float(rp[i]), self._angle_map)
                true, _ = A.snap_to_class(float(hd[i]), float(rd[i]), self._angle_map)
                snap_ok += int(snap == true)
            total += x.size(0)
        total = max(1, total)
        return {"heading_err_deg": h_err / total, "roll_err_deg": r_err / total,
                "snap_acc": snap_ok / total}

    def _save(self, path: Path, config) -> None:
        torch.save({
            "backbone": config.backbone, "imgsz": config.imgsz,
            "angle_map": self._angle_map, "state_dict": self._net.state_dict(),
        }, path)

    def load(self, weights_path: Path) -> None:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        self._angle_map = ckpt["angle_map"]
        self._imgsz = ckpt["imgsz"]
        self._net = _TwoAngleNet(ckpt["backbone"], pretrained=False)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        from types import SimpleNamespace
        self._eval_tf = _transforms(SimpleNamespace(imgsz=self._imgsz), train=False)

    @torch.no_grad()
    def predict(self, image_paths: Iterable[Path | str]) -> list[tuple[str, float]]:
        if self._net is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        out = []
        for p in image_paths:
            x = self._eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(self._device)
            hv, rv = self._net(x)
            heading = float(A.vec_to_deg(hv)[0])
            roll = float(A.vec_to_deg(rv)[0])
            name, conf = A.snap_to_class(heading, roll, self._angle_map)
            out.append((name, conf))
        return out

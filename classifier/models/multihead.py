"""
Factorized classifier: one shared backbone, two heads.

A composite label `<pose>_facing_<dir>` is the product of two near-independent
factors. Predicting them with separate heads turns the steep 16-way composite
imbalance into a much milder 8-way (pose) and 4-way (facing) problem, and lets
horizontal-flip augmentation be used (the flip is relabeled, see data.py).

model_type: "multihead". Consumes the on-disk split produced by
`python -m classifier.data`, so it shares train/val/test with every other model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from .base import Classifier
from .. import data as D
from .. import labels as L
from ..losses import build_loss, class_balanced_weights, counts_for

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in ("cpu",):
        print(f"  device {requested!r} unavailable, falling back to cpu")
    return torch.device("cpu")


def _build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    factory = getattr(torchvision.models, name, None)
    if factory is None:
        raise ValueError(f"Unknown torchvision backbone {name!r}")
    # At load() time the checkpoint overwrites these weights, so skip the
    # ImageNet download (also lets inference run offline).
    net = factory(weights="DEFAULT" if pretrained else None)
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif hasattr(net, "classifier"):
        head = net.classifier
        last = head[-1] if isinstance(head, nn.Sequential) else head
        feat_dim = last.in_features
        if isinstance(head, nn.Sequential):
            head[-1] = nn.Identity()
        else:
            net.classifier = nn.Identity()
    else:
        raise ValueError(f"Don't know how to strip the head of {name!r}")
    return net, feat_dim


class _TwoHeadNet(nn.Module):
    def __init__(self, backbone_name: str, n_pose: int, n_facing: int,
                 dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.pose_head = nn.Linear(feat_dim, n_pose)
        self.facing_head = nn.Linear(feat_dim, n_facing)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.pose_head(f), self.facing_head(f)


def _transforms(cfg, train: bool):
    from torchvision import transforms as T
    common = [T.Resize((cfg.imgsz, cfg.imgsz))]
    if train:
        a = cfg.aug
        ops = common + [
            T.RandomAffine(
                degrees=a.degrees,
                translate=(a.translate, a.translate) if a.translate else None,
                scale=(max(0.1, 1 - a.scale), 1 + a.scale) if a.scale else None,
            ),
            T.ColorJitter(brightness=a.hsv_v, saturation=a.hsv_s, hue=a.hsv_h),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
        if a.erasing:
            ops.append(T.RandomErasing(p=a.erasing))
        return T.Compose(ops)
    return T.Compose(common + [T.ToTensor(),
                               T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])


def _macro_f1(truths: list[str], preds: list[str]) -> float:
    from collections import Counter
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(truths, preds):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    classes = set(truths)
    total = 0.0
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        total += 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return total / max(1, len(classes))


class MultiHeadClassifier(Classifier):
    def __init__(self):
        self._net: _TwoHeadNet | None = None
        self._vocab: D.Vocab | None = None
        self._device = torch.device("cpu")
        self._imgsz = 224
        self._eval_tf = None
        self.tta = False
        self.abstain_threshold = 0.0

    @property
    def class_names(self) -> list[str]:
        if self._vocab is None:
            raise RuntimeError("Model not trained or loaded.")
        return self._vocab.composites

    def _loss_weights(self, labels: list[str], vocab: list[str], cfg):
        if cfg.loss == "ce":
            return None
        w = class_balanced_weights(counts_for(labels, vocab), cfg.cb_beta)
        return w.to(self._device)

    def train(self, config, output_dir: Path) -> Path:
        torch.manual_seed(config.seed)
        self._device = _resolve_device(config.device)
        self._imgsz = config.imgsz

        train_s = D.load_split_from_dir(config.data_dir, "train")
        val_s = D.load_split_from_dir(config.data_dir, "val")
        if not train_s:
            raise SystemExit(
                f"No training crops under {config.data_dir}/train. "
                f"Run: python -m classifier.data --out {config.data_dir} "
                f"--skip-class unclear")
        # Vocab spans every split so a factor seen only in val/test never throws
        # an index error. A factor with no train examples just stays unlearned.
        all_s = train_s + val_s + D.load_split_from_dir(config.data_dir, "test")
        vocab = D.build_vocab(all_s)
        self._vocab = vocab

        train_ds = D.FishCropDataset(train_s, vocab, _transforms(config, True),
                                     flip_p=config.aug.fliplr, rng_seed=config.seed)
        val_ds = D.FishCropDataset(val_s, vocab, _transforms(config, False))
        val_loader = DataLoader(val_ds, batch_size=config.batch, shuffle=False,
                                num_workers=config.num_workers)

        self._net = _TwoHeadNet(config.backbone, len(vocab.poses),
                                len(vocab.facings), dropout=config.dropout).to(self._device)

        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "best.pt"

        def loader(balanced: bool) -> DataLoader:
            sampler = D.weighted_sampler(train_s, config.seed) if balanced else None
            return DataLoader(train_ds, batch_size=config.batch, sampler=sampler,
                              shuffle=sampler is None, num_workers=config.num_workers,
                              drop_last=False)

        def losses(class_balanced: bool):
            if class_balanced:
                pw = class_balanced_weights(counts_for([s.pose for s in train_s],
                                                       vocab.poses), config.cb_beta).to(self._device)
                fw = class_balanced_weights(counts_for([s.facing for s in train_s],
                                                       vocab.facings), config.cb_beta).to(self._device)
                kind = "class_balanced"
            else:
                pw = fw = None
                kind = "ce"
            return (build_loss(kind, pw, config.focal_gamma, config.label_smoothing).to(self._device),
                    build_loss(kind, fw, config.focal_gamma, config.label_smoothing).to(self._device))

        def set_backbone_trainable(flag: bool) -> None:
            for p in self._net.backbone.parameters():
                p.requires_grad = flag

        state = {"best_f1": -1.0, "best_epoch": -1, "global_epoch": 0, "history": []}

        def run_phase(label, n_epochs, dl, pose_loss, facing_loss, opt, sched,
                      backbone_mode, use_mixup, save_best, early_stop):
            mix = (torch.distributions.Beta(config.aug.mixup, config.aug.mixup)
                   if use_mixup and config.aug.mixup > 0 else None)
            phase_best = -1.0
            phase_best_epoch = state["global_epoch"]
            for _ in range(n_epochs):
                ep = state["global_epoch"]
                if backbone_mode == "schedule":
                    set_backbone_trainable(ep >= config.freeze_epochs)
                    frozen = ep < config.freeze_epochs
                elif backbone_mode == "frozen":
                    frozen = True
                else:
                    frozen = False
                self._net.train()
                if frozen:
                    self._net.backbone.eval()  # keep BN stats fixed while frozen
                running = 0.0
                for x, yp, yf in dl:
                    x, yp, yf = x.to(self._device), yp.to(self._device), yf.to(self._device)
                    opt.zero_grad()
                    if mix is not None:
                        lam = mix.sample().item()
                        perm = torch.randperm(x.size(0), device=self._device)
                        lp, lf = self._net(lam * x + (1 - lam) * x[perm])
                        loss = (lam * (pose_loss(lp, yp) + facing_loss(lf, yf))
                                + (1 - lam) * (pose_loss(lp, yp[perm]) + facing_loss(lf, yf[perm])))
                    else:
                        lp, lf = self._net(x)
                        loss = pose_loss(lp, yp) + facing_loss(lf, yf)
                    loss.backward()
                    opt.step()
                    running += loss.item() * x.size(0)
                sched.step()

                metrics = self._evaluate(val_loader, vocab)
                state["history"].append({"epoch": ep, "phase": label,
                                         "train_loss": running / max(1, len(train_ds)),
                                         **metrics})
                f1 = metrics["composite_macro_f1"]
                flag = ""
                if save_best and f1 > state["best_f1"]:
                    state["best_f1"] = f1
                    state["best_epoch"] = ep
                    self._save(weights_path, config)
                    flag = "  *"
                if f1 > phase_best:
                    phase_best = f1
                    phase_best_epoch = ep
                print(f"  [{label}] epoch {ep:>3}  loss {running/max(1,len(train_ds)):.3f}  "
                      f"pose_acc {metrics['pose_acc']:.3f}  facing_acc {metrics['facing_acc']:.3f}  "
                      f"comp_acc {metrics['composite_acc']:.3f}  "
                      f"comp_macroF1 {f1:.3f}{flag}")
                state["global_epoch"] += 1
                if early_stop and ep - phase_best_epoch >= config.patience:
                    print(f"  early stop (no val gain for {config.patience} epochs)")
                    break

        if config.decouple_epochs > 0:
            # Stage 1 — representation: natural sampling, plain CE, full backbone.
            pl, fl = losses(class_balanced=False)
            opt = torch.optim.AdamW(self._net.parameters(), lr=config.lr0,
                                    weight_decay=config.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
            run_phase("stage1-repr", config.epochs, loader(balanced=False), pl, fl,
                      opt, sched, backbone_mode="full", use_mixup=True,
                      save_best=False, early_stop=False)
            # Stage 2 — classifier: reinit heads, freeze backbone, balanced + CB loss.
            self._net.pose_head.reset_parameters()
            self._net.facing_head.reset_parameters()
            set_backbone_trainable(False)
            pl, fl = losses(class_balanced=True)
            head_params = list(self._net.pose_head.parameters()) + list(self._net.facing_head.parameters())
            opt = torch.optim.AdamW(head_params, lr=config.lr0, weight_decay=config.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.decouple_epochs)
            run_phase("stage2-clf", config.decouple_epochs, loader(balanced=True), pl, fl,
                      opt, sched, backbone_mode="frozen", use_mixup=False,
                      save_best=True, early_stop=True)
        else:
            pose_w = self._loss_weights([s.pose for s in train_s], vocab.poses, config)
            facing_w = self._loss_weights([s.facing for s in train_s], vocab.facings, config)
            pl = build_loss(config.loss, pose_w, config.focal_gamma, config.label_smoothing).to(self._device)
            fl = build_loss(config.loss, facing_w, config.focal_gamma, config.label_smoothing).to(self._device)
            opt = torch.optim.AdamW(self._net.parameters(), lr=config.lr0,
                                    weight_decay=config.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
            run_phase("train", config.epochs, loader(balanced=config.weighted_sampler),
                      pl, fl, opt, sched, backbone_mode="schedule", use_mixup=True,
                      save_best=True, early_stop=True)

        (output_dir / "multihead_report.json").write_text(json.dumps({
            "best_epoch": state["best_epoch"],
            "best_composite_macro_f1": state["best_f1"],
            "n_train": len(train_s), "n_val": len(val_s),
            "two_stage": config.decouple_epochs > 0,
            "poses": vocab.poses, "facings": vocab.facings,
            "history": state["history"],
        }, indent=2))

        if not weights_path.exists():
            raise RuntimeError("Training produced no checkpoint.")
        return weights_path

    @torch.no_grad()
    def _evaluate(self, loader, vocab: D.Vocab) -> dict:
        self._net.eval()
        pose_ok = facing_ok = comp_ok = total = 0
        comp_t, comp_p = [], []
        for x, yp, yf in loader:
            x = x.to(self._device)
            lp, lf = self._net(x)
            pp = lp.argmax(1).cpu()
            pf = lf.argmax(1).cpu()
            pose_ok += (pp == yp).sum().item()
            facing_ok += (pf == yf).sum().item()
            for i in range(x.size(0)):
                t = L.compose(vocab.poses[yp[i]], vocab.facings[yf[i]])
                p = L.compose(vocab.poses[pp[i]], vocab.facings[pf[i]])
                comp_t.append(t)
                comp_p.append(p)
                comp_ok += int(t == p)
            total += x.size(0)
        total = max(1, total)
        return {
            "pose_acc": pose_ok / total,
            "facing_acc": facing_ok / total,
            "composite_acc": comp_ok / total,
            "composite_macro_f1": _macro_f1(comp_t, comp_p),
        }

    def _save(self, path: Path, config) -> None:
        torch.save({
            "backbone": config.backbone,
            "imgsz": config.imgsz,
            "poses": self._vocab.poses,
            "facings": self._vocab.facings,
            "composites": self._vocab.composites,
            "tta": config.tta,
            "abstain_threshold": config.abstain_threshold,
            "state_dict": self._net.state_dict(),
        }, path)

    def load(self, weights_path: Path) -> None:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        self._vocab = D.Vocab(ckpt["poses"], ckpt["facings"], ckpt["composites"])
        self._imgsz = ckpt["imgsz"]
        self._net = _TwoHeadNet(ckpt["backbone"], len(self._vocab.poses),
                                len(self._vocab.facings), pretrained=False)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        from types import SimpleNamespace
        self._eval_tf = _transforms(SimpleNamespace(imgsz=self._imgsz), train=False)

        self.tta = ckpt.get("tta", False)
        self.abstain_threshold = ckpt.get("abstain_threshold", 0.0)
        # Re-read the run config if present so tta/abstain can be retuned without
        # retraining — they only affect inference, not the weights.
        cfg_path = Path(weights_path).resolve().parents[1] / "config.yaml"
        if cfg_path.exists():
            from ..config import load_config
            cfg = load_config(cfg_path)
            self.tta = cfg.tta
            self.abstain_threshold = cfg.abstain_threshold

        # Index maps to fold a horizontally-flipped view's probs back into the
        # original frame (the flip swaps left<->right in both factors).
        v = self._vocab
        self._pose_flip = torch.tensor([v.pose_idx(L.flip_pose_lr(p)) for p in v.poses])
        self._facing_flip = torch.tensor([v.facing_idx(L.flip_facing_lr(f)) for f in v.facings])

    @torch.no_grad()
    def _probs(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._eval_tf(img).unsqueeze(0).to(self._device)
        lp, lf = self._net(x)
        return torch.softmax(lp, 1)[0].cpu(), torch.softmax(lf, 1)[0].cpu()

    @torch.no_grad()
    def embed_and_predict(self, image_paths: list[Path | str], batch_size: int = 64):
        """Batched: return (embeddings [N,D] L2-normalized, pose_probs [N,P],
        facing_probs [N,F]) for candidate mining. Skips unreadable images."""
        if self._net is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        embs, poses, facings, ok = [], [], [], []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors, kept = [], []
            for p in batch_paths:
                try:
                    tensors.append(self._eval_tf(Image.open(p).convert("RGB")))
                    kept.append(p)
                except (OSError, ValueError):
                    continue
            if not tensors:
                continue
            x = torch.stack(tensors).to(self._device)
            feat = self._net.backbone(x)
            lp, lf = self._net.pose_head(feat), self._net.facing_head(feat)
            embs.append(torch.nn.functional.normalize(feat, dim=1).cpu())
            poses.append(torch.softmax(lp, 1).cpu())
            facings.append(torch.softmax(lf, 1).cpu())
            ok.extend(kept)
        if not ok:
            return [], torch.empty(0), torch.empty(0), torch.empty(0)
        return ok, torch.cat(embs), torch.cat(poses), torch.cat(facings)

    @torch.no_grad()
    def predict(self, image_paths: Iterable[Path | str]) -> list[tuple[str, float]]:
        if self._net is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        out: list[tuple[str, float]] = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            pose_acc = torch.zeros(len(self._vocab.poses))
            facing_acc = torch.zeros(len(self._vocab.facings))
            views: list[tuple[Image.Image, bool]] = [(img, False)]
            if self.tta:
                views += [(img.transpose(Image.FLIP_LEFT_RIGHT), True),
                          (img.rotate(10), False), (img.rotate(-10), False)]
            for view, flipped in views:
                pp, pf = self._probs(view)
                if flipped:
                    pp, pf = pp[self._pose_flip], pf[self._facing_flip]
                pose_acc += pp
                facing_acc += pf
            pose_acc /= len(views)
            facing_acc /= len(views)
            pi = int(pose_acc.argmax())
            fi = int(facing_acc.argmax())
            conf = float(pose_acc[pi] * facing_acc[fi])
            if self.abstain_threshold and conf < self.abstain_threshold:
                out.append(("unclear", conf))
            else:
                out.append((L.compose(self._vocab.poses[pi], self._vocab.facings[fi]), conf))
        return out

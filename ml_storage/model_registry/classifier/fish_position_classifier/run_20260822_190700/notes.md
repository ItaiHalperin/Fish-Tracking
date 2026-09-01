# roll_cls_angular — 2026-08-21

Ablation of `roll_cls`: identical config except the loss. Flat class-balanced
cross-entropy is replaced by a distance-aware soft target (tau = 45°), so a
mistake costs what it is worth in degrees — belly_down↔flank (90°) costs half of
belly_down↔belly_up (180°). Labels are unchanged: still the same class folders.

Early-stopped at epoch 18, best epoch 5 (val macro F1 0.916, val upside-down 100%).

## Test set (266 crops, same split as the baseline)

| | roll_cls | roll_cls_angular | Δ |
|---|---|---|---|
| Roll accuracy | 91.73% | **93.61%** | +1.88 |
| Macro F1 | 0.861 | **0.903** | **+0.042** |
| Roll angular error (mean) | 8.8° | **6.8°** | −2.0° |
| Upside-down accuracy | 98.12% | 97.74% | −0.38 |

Per class F1: right_flank 0.69 → **0.83** (n=12), left_flank 0.85 → 0.86 (n=46),
belly_down 0.94 → 0.96 (n=137), belly_up 0.97 → 0.96 (n=71).

## Reading

The +0.042 macro F1 comes from the flanks, which is where the mechanism predicts
it should. `right_flank` was the weakest class in the project; the soft target
encodes that a flank is *nearer* to belly_down than belly_up is, an ordering flat
cross-entropy cannot express. For scale this is comparable to what decoupled cRT
bought on the composite task (+0.035), from one loss term and no new labels.

Caveat, not to be buried: **upside-down accuracy fell 98.12% → 97.74%** — one crop
of 266, noise rather than trend, but it is the project's headline metric. belly_up
F1 likewise moved 0.97 → 0.96. The trade is that the headline metric is unchanged
within noise while overall roll quality and the weak classes improve materially.

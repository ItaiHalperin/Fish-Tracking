# Training run index

One row per training run. After running `python -m classifier.train --config <c>`
and `python -m classifier.test --run results/<dir>`, append a row here by hand.

| Run dir | Model | Aug notes | Train / Val / Test | Acc | Macro F1 | Notes |
|---|---|---|---|---|---|---|
| multihead_default_20260619_113646 | multihead (resnet34) | flip+remap, deg10, erase0.3, CB-loss, wsampler | 972 / 333 / 231 | 79.65% | 0.762 | First factorized run (split: old `labels`, now lost). Pose head strong (6 pose-only errs); facing weaker (down↔l/r). 28/47 errors both heads wrong = ambiguous crops. |
| multihead_reg_20260619_171417 | multihead (resnet34) | + dropout0.3, mixup0.2, ls0.05, wd5e-4, freeze5 | 972 / 333 / — | — | val 0.769 | Over-regularized: val macroF1 < baseline 0.799, train loss stuck ~1.1 (underfit). Test split lost (webapp overwrote `labels/`). Superseded by labels_split + moderated config. |
| multihead_default_20260619_232532 | multihead (resnet34) | flip+remap, deg10, erase0.3, CB-loss, wsampler | 1139 / 375 / 268 | 80.60% | 0.838 | New baseline on `labels_split` (more data). Macro F1 +0.076 vs first run — extra data helped the tail. |
| multihead_reg_20260620_083559 | multihead (resnet34) | + dropout0.2, ls0.05, mixup0.1, wd2e-4, freeze3, TTA | 1139 / 375 / 268 | 82.84% | 0.834 | Moderate reg + TTA. Acc +2.2pt but macro F1 tied (−0.004). Confirms data/label ceiling, not a tuning ceiling. |
| multihead_decoupled_20260620_152123 | multihead (resnet34) | Option 2: cRT two-stage (s1 natural+CE, s2 frozen heads balanced+CB), TTA | 1139 / 375 / 268 | 87.31% | 0.873 | **Current best.** +0.035 macro F1 vs baseline — decoupling fixed the classifier's tail bias. Weakest real classes now ~0.86 F1. |
| multihead_decoupled_20260620_175813 | multihead (resnet34) | Option 2 cRT, post-mining data, class-aware split | 1521 / 363 / 266 | 83.08% | 0.718 | After 2 mining rounds + class-aware split fix. NOT comparable to 0.873 (different split, test now includes harder rare classes — 14 vs 13). Macro F1 dragged by thin-support tail classes in test. |

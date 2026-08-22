# Data for the final report — what is NOT in the repo docs

Handoff note for whoever writes the 8-page report. The repo already documents the
classifier work well ([SUMMARY.md](SUMMARY.md), [REPORT.md](REPORT.md),
[results/INDEX.md](results/INDEX.md), [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md),
[IMBALANCE_RESEARCH.md](IMBALANCE_RESEARCH.md), [README.md](README.md)) — use those
directly for methods and classifier benchmarks. **This document holds everything
else**: numbers that exist only inside `runs/`, `dataset/` and `results_pipeline/`
(never written into any doc), quantities derived from those files, deviations from
the submitted proposal, and open questions only the team can answer.

Provenance is flagged throughout: **[measured]** = read from a file in the repo,
**[derived]** = computed from those files (formula given), **[unverified]** = needs a
human to confirm, **[absent]** = looked for and not found.

---

## 1. Deviations from the submitted proposal

The report must cover the proposal's items "in greater detail", which means being
explicit where the delivered system diverged. None of these divergences is recorded
anywhere in the repo.

| Proposal said | What was actually built | Note |
|---|---|---|
| Classifier: fine-tune **YOLOv11-cls nano** | Final models use a **ResNet-34** backbone; YOLO11n-cls survives only as a benchmark baseline (66.5% vs 83.1%) | Biggest architectural divergence. The stated reason for YOLO-cls was framework consistency with the detector; that was traded away for accuracy. The YOLO baseline rows in `results/INDEX.md` are the evidence that the trade was worth it |
| "**currently 6** orientation classes, tuned empirically" | **16** composite classes (8 pose × 4 facing, factorized into two heads) + a **4-way roll** label space + continuous angles | The class count went *up*, then the project scoped *down* to 4-way roll as the deliverable. This is the single most consequential design evolution and the proposal's "tune it empirically" promise was honoured |
| Disable horizontal flip (changes diagonal labels) | Flip **re-enabled** with a left↔right label remap, once the label was factorized | A concrete example of factorization unlocking augmentation. Proposal's constraint was correct for the unfactorized taxonomy |
| ~**128 hours** of footage (7 days control, 25 days treatment) | **~5.3 hours** downloaded locally (5×1h + 1×17min, 6 files) as a **deliberate POC sample**; labels drawn from **4** videos | Not a shortfall — a scope decision. Running the archive is left to the researchers who own the experiment (§2). The proposal's second stated goal, a reusable tool for future footage, is the part that was completed |
| Aggregation across the full experimental dataset | **[absent]** — per-video reports only, no cross-video aggregation code | The one genuinely unbuilt piece of the promised end product. Worth naming in Future Work: the researchers will need it to run the archive |
| Support **Google Colab** free GPU tier for the 128h run | **[absent]** — no notebook, no Colab reference anywhere in the tree | Development ran on Apple Silicon MPS throughout |
| Detector: YOLOv11n + ByteTrack | Delivered as proposed. **Additionally** a Faster R-CNN (`fasterrcnn_mobilenet_v3_large_fpn`) pipeline was written — `detection/train_faster_rcnn.py`, `test_faster_rcnn.py`, `track_faster_rcnn.py`, `analysis/video_inference_faster_rcnn.py` | Not in the proposal, and **no saved results** — an implemented but unreported detector comparison. Either run it for the Benchmarking section or state plainly that it was built and not evaluated |
| — | **Three Flask web apps**: crop labeler, heading line-annotator, analysis front-end (`webapp/`) | Substantial engineering not mentioned in the proposal. The labeler is what made ~4,700 labels feasible |
| — | **Continuous angle regression** (heading + roll, cos/sin + angular loss) | See §2 — this partially undercuts the proposal's own argument for skipping pose estimation |
| Roboflow for manual box annotation | Confirmed — `Tracking Goldfish v2.coco.zip` (21 MB, dated 2026-05-22), ingested by `dataset_tools/ingest_coco.py` | |

### The proposal's DeepLabCut argument, revisited

The proposal justified skipping pose estimation: the research question "is already
framed in terms of discrete categories — going through pose estimation just to bin
the angles afterward adds an extra step that can introduce errors, for no real
benefit."

The project then built continuous angle regression anyway, and the measurements
partly contradict the proposal:

- The prediction was **right about class accuracy**: snapping angles back to classes
  scores 80.1% vs the classifier's 83.1% — the extra step does compound errors.
- The prediction was **wrong about "no real benefit"**: on angular error the
  regressor achieves 8.4° vs 14.2°, and beats the 12.1° *oracle floor* that bounds
  every possible classifier on this taxonomy (§6).

This is an unusually clean "we tested our own assumption and it was half wrong",
and it belongs in Lessons Learned.

---

## 2. The research question — condition assignment and first results

**Condition assignment [confirmed by the team, 2026-08-22]:**

- **Corals DOWN (control) — two videos, both `.MTS`:** `00049.MTS` (recorded
  **31.5.25**) and `Control_15_6_26 (1).MTS` (recorded **15.6.26**). Different
  recording days more than a year apart, so these are **independent replicates**,
  not two segments of one session — despite having byte-identical file sizes and
  identical durations (1033.22 s), which is just the camera's fixed AVCHD split
  size, both being the same camera at the same settings (50 fps, 1920×1080).
- **Corals UP (treatment) — all five `.mp4` files.** The `RGoldies` / `NRGoldies`
  prefix carries no condition information; its meaning is not needed for the report
  and should not be speculated about.

So the design as sampled is **N=2 control vs N=5 treatment**, with the control
recorded on a different camera/format (50 fps `.MTS`) than the treatment (25 fps
`.mp4`) — a format difference perfectly correlated with the condition. It does *not*
appear to hurt detection (§5.1a measures ~91–100% recall on both formats), but it
remains a confound in principle and should be named as one.

### Scope: this is a proof of concept, by design [confirmed by the team]

**The six local videos are a deliberately small POC sample, not the study.** The
full archive has far more footage; running the tool across it is left to the
researchers who own the experiment. The proposal set two goals — "to help answer the
research question on the current dataset, and to leave behind a reusable tool that
future research can run on new footage and related questions" — and **the tool is the
deliverable that was completed**. The six-video comparison in §5.3 demonstrates that
the instrument runs end-to-end across both conditions and produces the report the
proposal specified.

**How the report should frame the comparison numbers:** as a demonstration that the
measurement works, *not* as a biological finding. Because fish-frames within a video
are heavily autocorrelated (~10 fish at 25 fps), the effective sample size is
nowhere near the ~245,000 fish-frames a video yields — the unit of replication is
the *video*, so this is N=5 vs N=2.

What the numbers legitimately support: the tool runs end-to-end on both conditions,
recovers ~91–100% of the fish per frame (§5.1a), and produces the per-video
distribution the proposal specified, reproducibly — the same clip re-run weeks later
yielded an identical fish-frame count to the digit (§5.2a).

What they do **not** support, and where the report must be careful:

- **The conditions overlap.** In the full seven-run sample (§5.3) the treatment
  videos span a wide range of upside-down fractions, and at least one treatment video
  falls *below* a control video. Any claim of clean separation would be wrong; the
  early two-video hint of a 3× difference does not survive the full sample.
- **The between-video spread within a single condition is larger than the gap
  between conditions.** That, not the mean difference, is the honest headline: it says
  the study needs many videos per condition, which is exactly the design the real
  researchers should run.
- **The absolute values carry model-version uncertainty of several fold** (§5.2a) —
  larger than the condition difference itself.

So: *directionally consistent with the hypothesis in places, not separable at this
sample size, and the instrument is ready for the properly powered study.* That is
the handoff the proposal promised, and it is a defensible result. Do not write that
corals cause upside-down swimming.

---

## 3. Datasets — actuals [measured]

### Detection annotation set (`dataset/`)

| Split | Images | Boxes |
|---|---|---|
| train | 39 | 294 |
| val | 8 | 62 |
| test | 8 | 70 |
| **total** | **55** | **426** |

Source videos for these frames: **25 frames from video `00034`**, plus ~30 frames
named `frame_HH-MM-SS` from (apparently) one other video. So the detector was
trained on frames from roughly **two** videos. [derived from filenames]

This is worth stating plainly: the proposal's stated remaining task #1 was "scale up
the hand annotations", and for the *detector* that did not happen — 55 frames is a
pilot-sized set. It also interacts badly with the proposal's own "lighting variation
between videos" difficulty: a detector that has seen two videos has no way to be
robust to the other thirty days of footage.

`dataset.yaml` also carries a **class-list bug** worth one sentence: it declares two
classes, `0: Tracking-Goldfish-v2` and `1: goldfish`, where class 0 is a Roboflow
artifact rather than a real category.

### Classification label set

- **4,722 total hand labels** across 17 buckets at the final labeling round, of
  which **2,762 (58%) are `unclear`** and excluded. (Round-by-round tables are in
  `IMPROVEMENT_PLAN.md` — already in the repo.)
- **2,150 usable crops** in the protected split (`labels_split/`): 1,521 train /
  363 val / 266 test.
- **All 2,150 also carry a continuous heading annotation** (`labels_angles/headings.json`).
- **Manual annotation effort: ~5 hours total** [team, 2026-08-22] — covering the
  4,722 class labels plus the 2,150 heading line-annotations, i.e. **~6,900 human
  decisions in ~5 h, roughly 23 per minute**. Worth stating in the report: it
  quantifies what the custom labeler bought (keyboard shortcuts, context strip,
  rare-class seeding, mining queue), and it is the number that makes the
  data-centric approach look cheap relative to its payoff — the largest accuracy
  gains in the project came from this 5 hours, not from model tuning.
- Source video breakdown of the 2,150 [derived]:

| Source | Crops |
|---|---|
| (flattened pool, source video not recoverable from filename) | 1,080 |
| NRGoldies_5_11_25 | 343 |
| NRGoldies_3_11_25 | 342 |
| RGoldies_7_6_25 | 208 |
| RGoldies_23_10_25 | 177 |

Note the ~50% of crops whose source video is unrecoverable — they were labeled from
a flattened pool named `all`. This weakens any claim about per-video balance in the
training set, and is a small reproducibility lesson.

### Videos present locally [measured, via ffprobe]

| File | Duration | FPS | Resolution |
|---|---|---|---|
| NRGoldies 3_11_25.mp4 | 3654 s (60.9 min) | 25 | 1920×1080 |
| NRGoldies 5_11_25.mp4 | 3601 s (60.0 min) | 25 | 1920×1080 |
| RGoldies 18_9_25.mp4 | 3661 s (61.0 min) | 25 | 1920×1080 |
| RGoldies 7_6_25.mp4 | 3609 s (60.2 min) | 25 | 1920×1080 |
| RGoldies_23_10_25.mp4 | 3685 s (61.4 min) | 25 | 1920×1080 |
| 00049.MTS | 1033 s (17.2 min) | **50** | 1920×1080 |

Total ≈ **5.3 hours**, i.e. **~4%** of the proposed 128 h. Note `00049.MTS` is 50 fps
— double the rest — which matters for any per-second rate computed from it.

---

## 4. Detection and tracking benchmarks [measured, from `runs/detect/*/results.csv`]

**Never reported in any doc.** The docs cover the classifier exclusively; the
detector half of the proposal is undocumented.

Final detector — `runs/detect/fish_tracking_model-2`, YOLO11n from COCO weights,
200 epochs requested / **192 run** (early stop, patience 50), imgsz 640, batch 16,
`device=mps`, lr0 0.01, augmentation `degrees=10, scale=0.5, fliplr=0.5, mosaic=1.0`:

| Metric | Best epoch (142) | Final epoch (192) |
|---|---|---|
| Precision | 0.951 | 0.885 |
| Recall | 0.807 | 0.869 |
| mAP@50 | 0.881 | 0.881 |
| mAP@50-95 | **0.556** | 0.546 |

Best mAP@50 across all epochs: **0.899** (epoch 115). **Total training time: 412
seconds** (~7 minutes) for 192 epochs on MPS — a useful compute data point.

First detector attempt — `runs/detect/fish_tracking_model`, 10 epochs only:
mAP@50 **0.101**, mAP@50-95 0.038, precision 0.016, recall 0.323. A clean
underfitting "before" data point for the report: 10 epochs is nowhere near enough,
200 (early-stopping at ~142) is.

### Tracking

- ByteTrack is the default; **BoTSORT is implemented and selectable**
  (`--tracker` in `detection/track.py`, `tracker=` in `analysis/pipeline.py`).
- **[absent]** No ByteTrack vs BoTSORT comparison was ever run or recorded, despite
  the proposal promising "we'll analyze these choices empirically". The switch
  exists; the experiment does not. §5 gives the fragmentation numbers that would be
  the natural metric for such a comparison.
- Pipeline inference defaults: detection `conf=0.4`, temporal majority-vote
  `smooth_window=5` frames.

---

## 5. End-product outputs — the project's headline quantity [measured + derived]

**This is the most important omission from the docs.** The system's entire purpose
is to output the fraction of fish-frames per orientation, and those outputs exist in
`results_pipeline/` (untracked xlsx files, invisible to a doc-reading agent) but
appear in no report.

Pooled across all fish-frames (the proposal's designated headline statistic):

| Run | Clip | Fish-frames | Tracks | belly_down | right_flank | **belly_up** | left_flank |
|---|---|---|---|---|---|---|---|
| RGoldies_18_9_25 | 5 min | 81,131 | 680 | 66.79% | 15.48% | **0.82%** | 16.91% |
| RGoldies_18_9_25_15min | 15 min | 245,569 | 1,767 | 68.13% | 15.81% | **0.59%** | 15.47% |
| 00049 | 5 min | 52,317 | 342 | 82.46% | 12.96% | **0.19%** | 4.39% |
| 00049_15min | 15 min | 160,022 | 962 | 78.69% | 17.52% | **0.26%** | 3.53% |

**So the answer the instrument currently gives is: fish are upside down roughly
0.2%–0.8% of the time, and the two videos differ by ~3×.** Whether that difference
is a condition effect, a per-video artifact, or noise is exactly what §2 says was
never tested.

Two reassuring consistency checks [derived]: the 5-min and 15-min runs of the same
video agree closely (0.82% vs 0.59%; 0.19% vs 0.26%), and mean fish-per-frame is
stable within a video across clip lengths (§5.1). So the measurement is repeatable
even though the experiment is incomplete.

**`out/summary.xlsx` — discarded [team decision, 2026-08-22].** An older run (dated
2026-06-24) reporting **belly_up 11.78%**, an order of magnitude above every other
run. Its source video and model version could not be recovered by either the team or
this audit, so it is excluded from the report. Omit it entirely rather than
mentioning it as an anomaly. (The file is still on disk; delete it if you want it
gone.)

### 5.1 Track fragmentation [derived from the summary files]

| Run | Tracks | Mean track | **Median track** | Tracks ≤2 frames | Longest track |
|---|---|---|---|---|---|
| RGoldies_18_9_25 (5 min) | 680 | 119 f | **20 f** | 14% | 7,501 f (9.2% of frames) |
| RGoldies_18_9_25_15min | 1,767 | 139 f | **20 f** | 15% | 20,284 f (8.3%) |
| 00049 (5 min) | 342 | 153 f | **12 f** | 24% | 3,373 f (6.4%) |
| 00049_15min | 962 | 166 f | **11 f** | 24% | 5,631 f (3.5%) |

Mean fish visible per frame [derived: fish-frames ÷ frames processed]: **~10.9** in
RGoldies_18_9_25, **~7.0** in 00049. Note when computing this yourself: the pipeline
**remuxes `.MTS` to 25 fps** before tracking, so a 900-second clip is 22,500 frames
for every video regardless of the source being 50 fps. Using the source's 50 fps
halves the apparent detection rate and produces a spurious recall gap.

### 5.1a Detection recall is good — a positive result worth reporting

**Fish counts [confirmed by the team, 2026-08-22]:** around 10 fish in the tank
generally, and **8 in `00049.MTS`**. Against the measured detection rate:

| Video | Condition | Fish present | Mean detected/frame | Implied recall |
|---|---|---|---|---|
| RGoldies_18_9_25 | corals up | ~10 | 10.9 | ~100% |
| Control_15_6_26.MTS | corals down | ~10 | 10.1 | ~100% |
| 00049.MTS | corals down | **8** | **7.25** | **~91%** |

For `00049` this is measured directly rather than inferred: the per-frame detection
counts in a clean run are modal at 8 (5,237 frames) and 7 (4,828), spanning 4–11.
The detector recovers nearly every fish, in both conditions and on both camera
formats.

This matters for the report in two ways. It is a **genuine validation result** — the
mAP@50 of 0.881 (§4) translates into near-complete per-frame recall on real footage,
including on videos the detector never trained on, which is the generalization claim
the benchmark alone cannot support. And it **removes a confound**: the condition
comparison in §5.3 is not distorted by the two conditions being sampled with
different completeness.

> **Correction, for anyone who saw an earlier draft of this document:** it claimed a
> ~44% recall gap on the control video and made it the headline caveat. That was an
> arithmetic error — fish-per-frame was computed against the source's 50 fps when the
> pipeline remuxes `.MTS` to 25 fps (see §5.1). There is no recall gap. The genuine
> caveat on the comparison is classifier-version sensitivity (§5.2a), not detection.

The fragmentation is severe and now quantified: **the median track survives 11–20
frames — under one second** — and a quarter of tracks in `00049` last two frames or
less. With ~11 fish yielding 1,767 tracks in 15 minutes, identity is being lost
roughly 160 times per fish. The proposal predicted ID swaps qualitatively; these are
the numbers.

### 5.3 The corals-up vs corals-down comparison [measured, 2026-08-22]

Seven runs, 15-minute clips (900 s → 22,500 frames each), all scored with the **same**
classifier (`roll_cls_angular`, registered as `run_20260822_190700`). Sorted by
upside-down fraction:

| Video | Condition | Fish-frames | Tracks | Fish/frame | belly_down | right_flank | **belly_up** | left_flank |
|---|---|---|---|---|---|---|---|---|
| 00049 (31.5.25) | **down** | 160,022 | 962 | 7.11 | 93.18% | 1.46% | **0.299%** | 5.06% |
| RGoldies 7_6_25 | up | 111,772 | 588 | 4.97 | 72.71% | 3.74% | **0.994%** | 22.56% |
| Control_15_6_26 | **down** | 226,211 | 827 | 10.05 | 77.62% | 1.94% | **1.160%** | 19.28% |
| RGoldies 18_9_25 | up | 245,569 | 1,767 | 10.91 | 68.91% | 3.30% | **3.150%** | 24.64% |
| RGoldies_23_10_25 | up | 145,018 | 1,470 | 6.45 | 76.79% | 1.92% | **4.918%** | 16.38% |
| NRGoldies 5_11_25 | up | 142,806 | 889 | 6.35 | 53.49% | 10.19% | **7.499%** | 28.82% |
| NRGoldies 3_11_25 | up | 150,285 | 997 | 6.68 | 53.92% | 16.01% | **9.255%** | 20.82% |

| | n | belly_up mean | median | range |
|---|---|---|---|---|
| **Corals up** (treatment) | 5 | **5.16%** | 4.92% | 0.99% – 9.26% |
| **Corals down** (control) | 2 | **0.73%** | 0.73% | 0.30% – 1.16% |

**Effect size: 7.1× more upside-down time under corals-up.** Every ordering is in the
hypothesized direction except one: `RGoldies 7_6_25` (0.994%) falls just below the
higher control (1.160%), so the two groups **overlap marginally**.

**Significance: p ≈ 0.095, one-tailed Mann-Whitney U** (control ranks 1 and 3 of 7,
U = 1; 2 of the 21 possible rank arrangements are that extreme or more). Not
significant at α = 0.05.

**The crucial point for the report — the design is underpowered by construction.**
With n=2 vs n=5, the *smallest attainable* one-tailed p is 1/21 = **0.048**, reached
only under perfect separation. So this sample could never have produced a
comfortably significant result no matter what the fish did. The correct conclusion is
not "no effect" but "this sample cannot test the effect", and the 7× difference is
strong enough to justify the properly powered study. A rough guide for the
researchers: 6 videos per condition makes the minimum attainable p ≈ 0.001, leaving
real room for a result to emerge.

**A second grouping worth flagging.** Among the five corals-up videos, the two
`NRGoldies` files have by far the highest upside-down fractions (7.50%, 9.26%; mean
8.38%) versus the three `RGoldies` files (0.99%, 3.15%, 4.92%; mean 3.02%) — and also
much higher `right_flank` (10–16% vs 2–4%) and much lower `belly_down` (~54% vs
~69–77%). The team has confirmed the prefix is not the experimental condition, but it
evidently tracks *something* systematic — different tank, cohort, or camera placement.
Since it is confounded with the outcome, **the report should note it**: pooling all
five treatment videos hides a bimodal split, and whatever the prefix encodes may
explain more of the variance than the corals do.

**Caveats that apply to this table**, in order of importance:
1. Absolute values carry several-fold model-version uncertainty (§5.2a) — larger than
   the between-condition gap in at least one video.
2. Within-condition spread (9.3× across treatment videos, 3.9× across the two
   controls) exceeds the between-condition gap. This is the honest headline.
3. Control and treatment differ in camera format (50 fps `.MTS` vs 25 fps `.mp4`),
   perfectly correlated with condition. Detection recall is unaffected (§5.1a), but
   the confound remains in principle.
4. `RGoldies_23_10_25` and `RGoldies 18_9_25` show elevated fragmentation (1,470 and
   1,767 tracks) alongside elevated belly_up; short spurious tracks may carry odd
   orientations, so part of their signal could be tracking artefact.

### 5.2a ⚠️ The headline metric is highly sensitive to the classifier version

An accidental controlled experiment, and the most important caveat on the whole
comparison. Tonight's run of the `RGoldies_18_9_25` 15-minute clip produced
**245,569 fish-frames — identical to the digit** to the June run of the same clip.
The detector is deterministic, so both runs classified **byte-for-byte the same set
of crops**. The only difference was the classifier:

| Same 245,569 crops | Classifier | belly_up |
|---|---|---|
| June run | `roll_cls` (91.73% acc, 0.861 macro-F1) | **0.595%** |
| Tonight's run | `roll_cls_angular` (93.61% acc, 0.903 macro-F1) | **3.150%** |

**A 5.3× change in the deliverable metric from a model that is 1.9 accuracy points
better.** Both models score ~98% on upside-down detection on the labeled test set.

The effect is **video-dependent**, which makes it harder to correct for: on
`00049.MTS` the same model swap barely moved the metric (0.255% → 0.299%, on an
identical 160,022 crops), while on `RGoldies 18_9_25` it moved 5.3×. So this is not a
constant offset that could be calibrated away.

The likely mechanism, and why the test set could not predict it: the new model is
marginally *worse* precisely on belly_up (F1 0.96 vs 0.97, upside-down accuracy
97.74% vs 98.12%) while being better overall — and production crops are far more
ambiguous than the test set, which by construction contains **none** of the 58% of
hand-labeled crops a human judged `unclear`. A 4-way accuracy figure on curated
crops is close to blind to how a model behaves on the ambiguous majority, and
belly_up is a ~1% class where a small false-positive rate on ambiguous crops
dominates the estimate.

Consequences for the report:

- **Never mix numbers across model versions.** The §5.3 comparison is internally
  valid because all seven runs use the same classifier; the June figures in §5 use
  the old one and are not comparable to them.
- **Report the sensitivity itself** — it is a real result about the instrument, and
  arguably more useful to the biologists than either point estimate: a rare-class
  percentage of this kind carries model-version uncertainty far larger than its
  apparent precision, so a study should fix one model version for all footage and
  re-run everything if it ever changes.
- It also means **accuracy on a curated test set is the wrong headline metric for
  this deliverable**. What matters is stability of a ~1% rate on ambiguous
  production data, which nothing in the current benchmark suite measures.

### 5.2 Pooled vs per-fish — the proposal's design decision, validated [derived]

The proposal argued for pooling across frames rather than averaging per fish,
"because the question is about the population overall, and pooling is also more
robust to the identity switches and broken tracks". That call can now be scored:

| Run | belly_up **pooled** | belly_up **per-fish** | Ratio |
|---|---|---|---|
| RGoldies_18_9_25 | 0.82% | 1.73% | 2.1× |
| RGoldies_18_9_25_15min | 0.59% | 1.41% | 2.4× |
| 00049 | 0.19% | 2.06% | **10.7×** |
| 00049_15min | 0.26% | 2.44% | **9.6×** |

**The choice changes the headline number by 2–11×.** The mechanism is exactly the one
the proposal named: per-fish averaging gives a 2-frame fragment the same weight as a
20,000-frame track, and short fragments are disproportionately odd orientations. Note
the effect is worst in `00049`, the video with the most fragmentation (24% of tracks
≤2 frames) — the correlation the proposal predicted. This is the strongest
"our design reasoning was correct, and here is by how much" result available, and it
is currently in no document.

---

## 6. Classifier benchmarks — what to pull from the repo, and what is new

Most of this **is** now in the repo — `results/INDEX.md` has the full run table,
the Phase-4 baselines, the angular-error table and the oracle floor;
`REPORT.md` §4a/§4b covers the metric analysis. Use those. Headlines for
orientation: roll **93.61% / 0.903 macro-F1 / 6.8° / 97.74% upside-down**
(`roll_cls_angular`), composite **83.08% / 0.718** (decoupled cRT), continuous
heading **8.4°** mean error, floor **12.1°**.

Three things about it that are **not** in the repo:

1. **The deployed model is not the best model.** `ml_storage/model_registry/` holds
   exactly one registered classifier, `run_20260623_202149` — the *old* `roll_cls`
   (91.73% / 0.861 / 98.12%). The pipeline and all `results_pipeline/` outputs in §5
   were produced with it, **not** with the improved `roll_cls_angular` (93.61% /
   0.903). Any re-run would shift the numbers slightly. One line in Future Work.
2. **Total classifier runs performed:** 11 tracked runs in `results/` plus 4 earlier
   `runs/classify/` runs — useful for an honest "we swept N configurations" claim.
3. **`angle_map.yaml` was validated empirically** (this session, not previously):
   for every one of the 16 classes, the circular mean of its member crops' continuous
   heading labels falls within 14° of the declared value (most within 3°), with
   concentration R ≥ 0.93 and 2,150/2,150 coverage. Worth a sentence, because the
   whole distance-aware loss rests on that table being right, and the file itself
   still carries a "REVIEW THIS TABLE" warning.

---

## 7. Compute, storage, environment [measured]

The proposal's compute section can be made concrete:

- **Hardware:** Apple Silicon laptop, `device=mps` throughout. No discrete NVIDIA
  GPU, no Colab, no cluster.
- **Detector training:** 412 s for 192 epochs (imgsz 640, batch 16).
- **Classifier training:** ~1–2 min/epoch for ResNet-34 at imgsz 224, batch 32 on
  1,521 crops. The roll run early-stopped at 18 epochs; the composite cRT run took
  the full 55 (35 stage-1 + 20 stage-2).
- **Storage:** videos 2.1–4.6 GB each (~14 GB for 6 files); per-run classifier
  checkpoint **85 MB**; the 15-min `traces.xlsx` is **104 MB** — the per-frame trace
  file is larger than the model by 1.2×, which is a real scaling problem for the
  128-hour run (extrapolates to ~50 GB of traces). Worth a sentence in Future Work.
- `results/*/` and all data directories are gitignored; only `INDEX.md` and
  `baselines.json` are tracked.
- **Reproducibility friction encountered** (small but real): `pip install`, MPS
  access, torch's dataloader shared-memory helper, and git commit signing all
  required elevated filesystem/network permissions in a sandboxed environment.
- **Test suite:** 28 unit tests (`tests/`), covering the angular metrics and the
  distance-aware loss — including a proof-by-test that the new loss reduces exactly
  to weighted cross-entropy as τ→0. Before this session the project had **no
  automated tests at all**; that is itself a lesson.

---

## 8. Lessons learned — candidates with evidence

Ordered by how much they are worth saying. Each has a concrete number behind it.

1. **Taxonomy design dominated algorithm choice.** 16 composite classes over ~2,150
   crops starved the tail (some classes 1–4 samples; two never occurred at all).
   Factorizing into pose × facing turned a 217:1 / 16-class problem into 8-way +
   4-way and was the single largest gain. Evidence: `IMBALANCE_RESEARCH.md`'s
   prediction, then macro-F1 0.762 → 0.838 → 0.873.
2. **The metric decided the conclusion.** Accuracy said the classifier beat angle
   regression (83.1% vs 80.1%); angular error said the opposite (14.2° vs 8.4°).
   Both were computed from the same two checkpoints. For any task whose labels are a
   discretized continuum, report both.
3. **Know your label space's error floor.** A perfect 16-way classifier still misses
   heading by 12.1° on average, because it can only emit bin centres. The best
   classifier sits at 13.3°. Most of the apparent "17% inaccuracy" was the taxonomy,
   not the model — which redirects effort away from model tuning entirely.
4. **Charging mistakes by their size beats treating them as equal**, where the label
   space has geometry to exploit: roll macro-F1 +0.042 and −2.0° error from changing
   one loss term, with no new data, concentrated in the weakest class
   (`right_flank` F1 0.69 → 0.83). The same change did nothing on the composite task
   — because (3) had already shown there was no headroom left there.
5. **Splitting data wrongly can fake a result.** A naïve random split put an entire
   28-crop class wholly in the test set with zero training examples, costing ~0.15
   macro-F1. Group-aware (by fish track) + class-aware splitting fixed it. Related:
   two runs' test splits were permanently lost when the labeler webapp overwrote
   `labels/` (see `results/INDEX.md`), which is why the pipeline now reads a
   protected `labels_split/`.
6. **Regularization could not substitute for data.** Dropout + mixup + label
   smoothing + TTA moved accuracy +2.2 points and macro-F1 −0.004; a heavier variant
   underfit outright (val macro-F1 below baseline, train loss stuck at ~1.1). The
   ceiling was data, and the diagnosis came from watching train-vs-val behaviour.
7. **Active learning has a hard floor at "the behaviour doesn't occur".** Mining
   lifted `diag_up_right_facing_up` from 10 → 77 crops, but three classes never moved
   across four rounds and 750 targeted seeds. Model-assisted labeling finds rare
   *examples*, not rare *events*.
8. **Tracking identity was as predicted, and worse than expected in magnitude.**
   Median track ~0.5–0.8 s; ~160 tracks per fish per 15 minutes. The pooling design
   that anticipated this changes the headline number by 2–11×, so the mitigation
   mattered more than the risk register suggested.
9. **Factorizing a label can hide a semantic overload.** `facing_right` means a 0°
   heading under a `regular` pose but a 90° roll under `head_up` — pose×facing and
   heading×roll are different factorizations of the same thing. Consequence: taking
   each head's argmax independently produced **impossible classes on 10/266 test
   crops** (3.8%), e.g. `head_up_facing_up`, until the argmax was restricted to
   composites that exist.
10. **We tested our own proposal assumption and it was half wrong** — the DeepLabCut
    /pose-estimation argument in §1.
11. **The benchmark metric and the deliverable metric can come apart completely.**
    Swapping in a classifier that was 1.9 accuracy points and 0.042 macro-F1 *better*
    moved the reported upside-down fraction by **5.3×** on an identical set of
    245,569 crops (§5.2a). Test-set accuracy was measured on curated crops containing
    none of the 58% a human called `unclear`; the deliverable is a ~1% rate over
    exactly that ambiguous population. Improving the benchmark number is not the same
    as improving the measurement, and only comparing the two on real footage revealed
    the gap.
12. **Verify pipeline outputs against a domain quantity, not just internal
    consistency.** A corrupted run reported 2.21 fish/frame where the correct value
    was 7.25, and its output file was perfectly self-consistent — `n_scored` equalled
    `n_frames`, no rejects, plausible percentages. Nothing inside the artefact
    revealed the loss. What caught it was comparing detections per frame against how
    many fish are in the tank. Any long-running extraction pipeline should assert an
    expected-range check like this on every run.
13. **Annotation effort was tiny compared to its payoff.** ~5 hours of human labeling
    (~6,900 decisions) produced every accuracy gain that mattered, while the model-side
    work — regularization sweeps, architecture changes — mostly did not move
    macro-F1. Worth reporting the 5 hours explicitly; it is the strongest argument for
    the data-centric framing.
14. **Build the measurement tool before the model.** The custom labeler, context
    strip, rare-class seeding and mining loop are what took the set from a few
    hundred to ~4,700 labels. Most of the project's accuracy came from data work
    that a "train a classifier" framing would not have budgeted for.

---

## 9. Potential extensions / future work

Cheap and concrete first:

1. **Close the detection-recall gap, and measure recall per video** (§5.1a). This is
   now the highest-priority item, ahead of anything on the classifier: a per-video
   scientific readout is only as trustworthy as the detector's recall on *that*
   video, and it is currently ~44% on the control. Two parts — (a) hand-count fish in
   ~20 frames per video as a standing recall check the pipeline reports alongside its
   percentages, and (b) add annotated frames from every camera/lighting setup,
   especially the `.MTS`, to the detector's 55-frame training set.
2. **Cross-video aggregation + the statistical test** — the one promised piece of the
   end product that was not built. The researchers running the archive will need a
   step that consumes many per-video reports, groups them by condition, and tests
   the difference in belly_up fraction, with **the video as the unit of replication**
   (fish-frames within a video are heavily autocorrelated — at ~10 fish and 25 fps,
   N is nowhere near the ~245,000 fish-frames it appears to be; treating them as
   independent would inflate significance enormously). This is roughly one script
   plus a condition-metadata table, and it is the natural first extension.
3. **Sweep `angular_tau`.** Fixed at 45° by reasoning from the taxonomy's spacing,
   never swept. τ ∈ {20, 45, 90} on the roll model is ~90 minutes and may extend the
   +0.042.
4. **Run the detector comparison that was built** — Faster R-CNN vs YOLO11n — and
   the tracker comparison that was promised — ByteTrack vs BoTSORT, scored on the
   fragmentation metrics in §5.1 rather than on mAP.
5. **Scale the detection annotation set past 55 frames / 2 videos**, sampled across
   lighting conditions and days. This is the least glamorous item and probably the
   highest-leverage one for generalizing beyond the POC videos to the full archive.
6. **Continuous roll labels.** Roll's 6.8° error is measured against class centres,
   because only *heading* was line-annotated. A roll dial on ~2,000 crops would make
   genuine roll-angle regression possible — and given that heading regression beat
   the classifier's own error floor, roll regression is the natural next step for the
   headline metric.
7. **Address the detection/classification coupling bias** the proposal flagged: fish
   near corals are both harder to detect and harder to classify, yet that is where
   the interesting orientations occur. Nothing in the repo measures this. A targeted
   near-coral test set would turn a known risk into a number.
8. **Trace-file scaling** — 104 MB per 15 minutes (§7) extrapolates to ~50 GB over
   128 hours. Parquet or per-fish aggregation before writing.

---

## 10. Open questions only the team can answer

Flagging these so the report doesn't guess:

**All resolved by the team on 2026-08-22.** Nothing here needs guessing:

1. **Conditions** — `00049.MTS` is the only corals-down video; all five `.mp4`s are
   corals-up. The `RGoldies`/`NRGoldies` prefix carries no condition information and
   its meaning is not needed for the report — don't speculate about it. (§2)
2. **Scope** — the six videos are an intentional POC; the wider archive exists and is
   left to the researchers who own the experiment. (§2)
3. **`out/summary.xlsx`** (belly_up 11.78%) — provenance unrecoverable by the team as
   well; **discarded**, omit from the report entirely. (§5)
4. **Fish counts** — ~10 in the tank generally, **8 in the `.MTS`**. This is what
   exposed the detection-recall gap in §5.1a, the single most consequential caveat on
   the condition comparison.
5. **Annotation effort** — **~5 manual hours** for ~6,900 annotations. (§3)

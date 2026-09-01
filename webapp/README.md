# Fish crop labeler

A tiny Flask app for hand-labeling fish crops into class folders, then balancing
and splitting them into the `labels/train` and `labels/val` layout that
the classifier pipeline expects.

## What it does

1. Reads unlabeled crops from a **source** folder (typically the output of
   `dataset_tools/flatten_crops.py`).
2. Shows them one at a time in the browser. You pick a class with a click or
   a number key. The file gets **moved** into `labels_raw/<class>/`.
3. When you're done, the **Finish** page shows per-class counts and lets you:
   - cap each class at a chosen max (random subsample of overflow goes to
     `labels_discarded/<class>/` for later inspection),
   - split the kept crops into `labels/train` and `labels/val` by a chosen
     validation ratio.

Re-finalizing is safe and idempotent: `labels/` and `labels_discarded/` get
wiped and rebuilt each time. Your raw labels in `labels_raw/` are never touched
after the labeling step, so you can re-cap with different numbers freely.

## Setup

From the repo root, in your existing venv:

```bash
pip install flask
```

That's the only new dependency.

## Prepare a source folder

The app expects a flat folder of crops (no subdirectories). Use the existing
`dataset_tools/flatten_crops.py` to produce one from `dataset_tools/extract_crops.py` output:

```bash
python -m dataset_tools.flatten_crops --crops-dir crops/RGoldies_23_10_25
# -> crops/RGoldies_23_10_25_flat/
```

## Run

```bash
python webapp/app.py --source crops/all_flat
```

Classes are auto-derived from the files in `webapp/static/reference/` (one
class per `<name>.png` / `<name>.jpg`), plus a final `unclear` bin that needs
no reference image. To override, pass `--classes a b c ...` explicitly.

To change the class taxonomy, just add or remove reference images in
`webapp/static/reference/`. Restart the app and the new classes appear.

Open <http://127.0.0.1:5050/> in a browser.

### Adding crops from another video

Just flatten and copy them into the same source folder before (or during)
labeling — the app re-scans each batch:

```bash
python -m dataset_tools.flatten_crops --crops-dir crops/AnotherVideo
cp crops/AnotherVideo_flat/* crops/RGoldies_23_10_25_flat/
```

## Labeling shortcuts

- **1–9**: assign the crop to the Nth class (in the order you passed `--classes`).
- **U**: undo the previous label (the file is moved back to source and re-shown).
- Click **Finish →** in the top right when you've labeled enough.

## Finishing

On the Finish page:

- The cap field defaults to the **smallest non-zero class count** so the
  classes start balanced. Raise it if you have plenty of every class, lower
  it to be more aggressive about balancing.
- Validation ratio default 0.2, test ratio default 0.1 (10% held out for
  benchmarking via `classifier.test`). Set test ratio to 0 if you don't
  want a separate test split.
- Click **Finalize**. You'll see a summary like:

  ```json
  {
    "regular_facing_right": {"total_labeled": 240, "kept": 100, "discarded": 140, "train": 70, "val": 20, "test": 10},
    "head_down_facing_left": {"total_labeled":  35, "kept":  35, "discarded":   0, "train": 25, "val":  7, "test": 3},
    ...
  }
  ```

After that:

```bash
python -m classifier.data --out labels_split --skip-class unclear
python -m classifier.train --config configs/roll_cls_angular.yaml
```

## Layout

```
crops/<video>_flat/             # source you point --source at
labels_raw/<class>/             # all human labels (source of truth)
labels/train/<class>/           # balanced + split, what training reads
labels/val/<class>/
labels_discarded/<class>/       # overflow from the cap, kept for review
```

## CLI reference

```
--source         Required. Flat folder of crops to label.
--classes        Required. Space-separated class names.
--raw-dir        Default: labels_raw
--out-dir        Default: labels (the labeler's own split; training reads labels_split/)
--discarded-dir  Default: labels_discarded
--host           Default: 127.0.0.1
--port           Default: 5050
```

## Notes & caveats

- Files are **moved**, not copied, into `labels_raw/` while labeling — this is
  what makes "unlabeled" a meaningful concept (= still in source). If you want
  to relabel a crop, the easiest path is to move it back into source manually
  and the app will surface it again.
- Undo only works for the **current server session** (it's an in-memory stack).
  Restart = no undo history.
- The app trusts that the classes you pass match the subfolders in
  `labels_raw/`. If you rerun with a different `--classes` list, existing class
  folders are not deleted — they're just ignored by the UI. Delete them by hand
  if you change your taxonomy.

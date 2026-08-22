"""
Score any orientation model in degrees, not just right/wrong.

Class accuracy treats a 45°-adjacent miss and a 180° miss as the same event,
which hides most of what a model gets wrong about orientation. This module
converts predictions to (heading, roll) degrees — via configs/angle_map.yaml for
a classifier, or straight from the head of a regressor — and reports how far off
they were.

Ground-truth heading prefers the continuous line annotations
(labels_angles/headings.json) over the class centre, which also makes the
**oracle floor** measurable: the error a perfect classifier still pays because it
can only ever emit a bin centre. Roll has no continuous labels, so its truth is
always the class centre.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from . import angles as A

THRESHOLDS = (15, 30, 45, 90)
UPSIDE_DOWN_TOLERANCES = (45, 60, 90)


def circular_error(a: float, b: float) -> float:
    """Smallest absolute difference on the circle, in degrees [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def roll_deg_from_name(name: str) -> float:
    for deg, roll_name in A.ROLL_NAMES.items():
        if roll_name == name:
            return float(deg)
    raise ValueError(f"Unknown roll class {name!r}")


def summarize(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    return {
        "n": len(errors),
        "mean_err_deg": mean(errors),
        "median_err_deg": median(errors),
        "within": {str(t): sum(e <= t for e in errors) / len(errors) for t in THRESHOLDS},
    }


def _per_class(truths: list[str], errors: list[float]) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for cls, err in zip(truths, errors):
        grouped[cls].append(err)
    return {cls: summarize(errs) for cls, errs in sorted(grouped.items())}


def angle_metrics(names: list[str], truths: list[str], preds: list[str] | None,
                  angle_map: dict[str, tuple[float, float]],
                  headings: dict[str, float],
                  pred_angles: list[tuple[float, float]] | None = None) -> dict:
    """Angular report for one model on one test set.

    `preds` are predicted composite class names (a classifier); pass
    `pred_angles` instead for a model that outputs continuous angles directly.
    `names` are crop filenames, used to look up continuous heading truth.
    """
    if pred_angles is None:
        if preds is None:
            raise ValueError("Provide either preds (class names) or pred_angles")
        pred_angles = [angle_map[p] for p in preds]

    true_centers = [angle_map[t] for t in truths]
    true_headings = [headings.get(n, center[0]) for n, center in zip(names, true_centers)]
    n_continuous = sum(1 for n in names if n in headings)

    heading_errs = [circular_error(p[0], t) for p, t in zip(pred_angles, true_headings)]
    roll_errs = [circular_error(p[1], t[1]) for p, t in zip(pred_angles, true_centers)]

    heading = summarize(heading_errs)
    heading["n_continuous_truth"] = n_continuous
    heading["per_true_class"] = _per_class(truths, heading_errs)

    roll = summarize(roll_errs)
    roll["per_true_class"] = _per_class(truths, roll_errs)
    roll["upside_down_accuracy"] = _upside_down_accuracy(pred_angles, true_centers)

    oracle = None
    if n_continuous:
        floor = [circular_error(center[0], t)
                 for center, t, n in zip(true_centers, true_headings, names)
                 if n in headings]
        oracle = {"heading": summarize(floor)}

    return {"heading": heading, "roll": roll, "oracle": oracle}


def _upside_down_accuracy(pred_angles: list[tuple[float, float]],
                          true_centers: list[tuple[float, float]]) -> dict:
    """Agreement on the binary call the project actually reports: is the fish
    belly-up? "Upside down" = roll within `tol` of 180°, swept because the
    threshold is a deliverable decision, not a fixed fact. 45° is the Voronoi
    boundary against the flanks at 90°/270° — the continuous equivalent of
    `pred == belly_up` — while wider tolerances count near-flank rolls as belly-up."""
    if not pred_angles:
        return {}
    out = {}
    for tol in UPSIDE_DOWN_TOLERANCES:
        ok = sum(int((circular_error(p[1], 180.0) <= tol) == (circular_error(t[1], 180.0) <= tol))
                 for p, t in zip(pred_angles, true_centers))
        out[str(tol)] = ok / len(pred_angles)
    return out


def format_report(report: dict, label: str = "") -> str:
    lines = [f"Angular error{' — ' + label if label else ''}:"]
    for axis in ("heading", "roll"):
        s = report[axis]
        if not s.get("n"):
            continue
        within = "  ".join(f"≤{t}° {s['within'][str(t)]*100:.0f}%" for t in THRESHOLDS)
        lines.append(f"  {axis:8s} mean {s['mean_err_deg']:5.1f}°  "
                     f"median {s['median_err_deg']:5.1f}°   {within}")
    ud = report["roll"]["upside_down_accuracy"]
    lines.append("  upside-down agreement: " +
                 "  ".join(f"±{t}° {ud[str(t)]*100:.2f}%" for t in UPSIDE_DOWN_TOLERANCES))
    if report.get("oracle"):
        o = report["oracle"]["heading"]
        lines.append(f"  oracle floor (perfect class -> centre): mean {o['mean_err_deg']:.1f}°  "
                     f"median {o['median_err_deg']:.1f}°  (n={o['n']})")
    return "\n".join(lines)

import math

import pytest

from classifier import angle_eval as AE


ANGLE_MAP = {
    "regular_facing_right": (0.0, 0.0),
    "regular_facing_left": (180.0, 0.0),
    "diag_up_right_facing_down": (45.0, 0.0),
    "upside_down_facing_right": (0.0, 180.0),
    "head_up_facing_right": (90.0, 90.0),
}


def test_circular_error_takes_the_short_way_round():
    assert AE.circular_error(359.0, 1.0) == pytest.approx(2.0)
    assert AE.circular_error(1.0, 359.0) == pytest.approx(2.0)
    assert AE.circular_error(0.0, 180.0) == pytest.approx(180.0)
    assert AE.circular_error(10.0, 10.0) == pytest.approx(0.0)


def test_circular_error_never_exceeds_180():
    for a in range(0, 360, 7):
        for b in range(0, 360, 11):
            assert 0.0 <= AE.circular_error(float(a), float(b)) <= 180.0


def test_circular_error_handles_out_of_range_degrees():
    assert AE.circular_error(-10.0, 350.0) == pytest.approx(0.0)
    assert AE.circular_error(730.0, 10.0) == pytest.approx(0.0)


def test_summarize_reports_mean_median_and_threshold_rates():
    s = AE.summarize([0.0, 10.0, 20.0, 100.0])
    assert s["n"] == 4
    assert s["mean_err_deg"] == pytest.approx(32.5)
    assert s["median_err_deg"] == pytest.approx(15.0)
    assert s["within"]["15"] == pytest.approx(0.5)
    assert s["within"]["30"] == pytest.approx(0.75)
    assert s["within"]["90"] == pytest.approx(0.75)


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert AE.summarize([])["n"] == 0


def test_roll_deg_from_name_inverts_roll_name():
    from classifier import angles as A

    for deg in (0.0, 90.0, 180.0, 270.0):
        assert AE.roll_deg_from_name(A.roll_name(deg)) == pytest.approx(deg)


def test_angle_metrics_scores_a_perfect_classifier_at_zero_error():
    names = ["a.jpg", "b.jpg"]
    truths = ["regular_facing_right", "upside_down_facing_right"]
    report = AE.angle_metrics(names, truths, truths, ANGLE_MAP, headings={})
    assert report["heading"]["mean_err_deg"] == pytest.approx(0.0)
    assert report["roll"]["mean_err_deg"] == pytest.approx(0.0)


def test_angle_metrics_charges_less_for_a_near_miss_than_a_far_one():
    truths = ["regular_facing_right"]
    near = AE.angle_metrics(["a.jpg"], truths, ["diag_up_right_facing_down"],
                            ANGLE_MAP, headings={})
    far = AE.angle_metrics(["a.jpg"], truths, ["regular_facing_left"],
                           ANGLE_MAP, headings={})
    assert near["heading"]["mean_err_deg"] == pytest.approx(45.0)
    assert far["heading"]["mean_err_deg"] == pytest.approx(180.0)


def test_angle_metrics_separates_heading_and_roll_errors():
    # Same heading, opposite roll: heading error 0, roll error 180.
    report = AE.angle_metrics(["a.jpg"], ["regular_facing_right"],
                             ["upside_down_facing_right"], ANGLE_MAP, headings={})
    assert report["heading"]["mean_err_deg"] == pytest.approx(0.0)
    assert report["roll"]["mean_err_deg"] == pytest.approx(180.0)


def test_continuous_headings_override_class_centers_as_truth():
    # The crop's real heading is 30°, not its class center of 0°.
    report = AE.angle_metrics(["a.jpg"], ["regular_facing_right"],
                             ["regular_facing_right"], ANGLE_MAP,
                             headings={"a.jpg": 30.0})
    assert report["heading"]["mean_err_deg"] == pytest.approx(30.0)
    assert report["heading"]["n_continuous_truth"] == 1
    # Roll has no continuous labels, so it is unaffected.
    assert report["roll"]["mean_err_deg"] == pytest.approx(0.0)


def test_oracle_floor_measures_taxonomy_loss_not_model_error():
    # A crop labeled regular_facing_right whose true heading is 20° off centre:
    # even a perfect classifier inherits that 20°.
    report = AE.angle_metrics(["a.jpg"], ["regular_facing_right"],
                             ["regular_facing_right"], ANGLE_MAP,
                             headings={"a.jpg": 20.0})
    assert report["oracle"]["heading"]["mean_err_deg"] == pytest.approx(20.0)


def test_oracle_floor_is_absent_without_continuous_labels():
    report = AE.angle_metrics(["a.jpg"], ["regular_facing_right"],
                             ["regular_facing_right"], ANGLE_MAP, headings={})
    assert report["oracle"] is None


def test_predicted_angles_may_be_supplied_directly_for_a_regressor():
    report = AE.angle_metrics(["a.jpg"], ["regular_facing_right"], None, ANGLE_MAP,
                              headings={"a.jpg": 30.0},
                              pred_angles=[(28.0, 5.0)])
    assert report["heading"]["mean_err_deg"] == pytest.approx(2.0)
    assert report["roll"]["mean_err_deg"] == pytest.approx(5.0)


def test_per_class_heading_error_is_broken_out():
    report = AE.angle_metrics(
        ["a.jpg", "b.jpg"],
        ["regular_facing_right", "regular_facing_left"],
        ["regular_facing_left", "regular_facing_left"],
        ANGLE_MAP, headings={})
    per = report["heading"]["per_true_class"]
    assert per["regular_facing_right"]["mean_err_deg"] == pytest.approx(180.0)
    assert per["regular_facing_left"]["mean_err_deg"] == pytest.approx(0.0)


def test_roll_metrics_include_the_upside_down_decision():
    # truth belly_up, predicted belly_down -> the upside-down call is wrong.
    report = AE.angle_metrics(["a.jpg", "b.jpg"],
                              ["upside_down_facing_right", "upside_down_facing_right"],
                              ["regular_facing_right", "upside_down_facing_right"],
                              ANGLE_MAP, headings={})
    for tol in AE.UPSIDE_DOWN_TOLERANCES:
        assert report["roll"]["upside_down_accuracy"][str(tol)] == pytest.approx(0.5)


def test_upside_down_tolerance_changes_where_a_flank_falls():
    # A true right flank (roll 90) predicted as belly_up (180): at ±45° the two
    # disagree; at ±90° the flank itself counts as upside down, so they agree.
    report = AE.angle_metrics(["a.jpg"], ["head_up_facing_right"],
                              ["upside_down_facing_right"], ANGLE_MAP, headings={})
    ud = report["roll"]["upside_down_accuracy"]
    assert ud["45"] == pytest.approx(0.0)
    assert ud["90"] == pytest.approx(1.0)

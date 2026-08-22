import pytest
import torch
import torch.nn as nn

from classifier.losses import AngularSoftTargetLoss, angular_cost_matrix, roll_cost_matrix


ANGLE_MAP = {
    "regular_facing_right": (0.0, 0.0),
    "diag_up_right_facing_down": (45.0, 0.0),
    "regular_facing_left": (180.0, 0.0),
    "upside_down_facing_right": (0.0, 180.0),
}
CLASSES = list(ANGLE_MAP)


def test_cost_matrix_is_zero_on_the_diagonal():
    cost = angular_cost_matrix(CLASSES, ANGLE_MAP)
    assert torch.allclose(cost.diagonal(), torch.zeros(len(CLASSES)))


def test_cost_matrix_is_symmetric():
    cost = angular_cost_matrix(CLASSES, ANGLE_MAP)
    assert torch.allclose(cost, cost.T)


def test_cost_matrix_sums_heading_and_roll_distance():
    cost = angular_cost_matrix(CLASSES, ANGLE_MAP)
    i = CLASSES.index("regular_facing_right")
    assert cost[i, CLASSES.index("diag_up_right_facing_down")] == pytest.approx(45.0)
    assert cost[i, CLASSES.index("regular_facing_left")] == pytest.approx(180.0)
    assert cost[i, CLASSES.index("upside_down_facing_right")] == pytest.approx(180.0)


def test_cost_matrix_axis_weights_are_applied():
    cost = angular_cost_matrix(CLASSES, ANGLE_MAP, heading_weight=0.0, roll_weight=1.0)
    i = CLASSES.index("regular_facing_right")
    # Heading ignored: the 45° heading neighbour now costs nothing...
    assert cost[i, CLASSES.index("diag_up_right_facing_down")] == pytest.approx(0.0)
    # ...while the roll-opposite class still costs 180°.
    assert cost[i, CLASSES.index("upside_down_facing_right")] == pytest.approx(180.0)


def test_roll_cost_matrix_is_circular():
    cost = roll_cost_matrix([0.0, 90.0, 180.0, 270.0])
    assert cost[0, 1] == pytest.approx(90.0)
    assert cost[0, 2] == pytest.approx(180.0)
    assert cost[0, 3] == pytest.approx(90.0)  # the short way round, not 270


def test_soft_targets_are_a_distribution_peaked_at_the_truth():
    loss = AngularSoftTargetLoss(angular_cost_matrix(CLASSES, ANGLE_MAP), tau=45.0)
    q = loss.soft_targets
    assert torch.allclose(q.sum(1), torch.ones(len(CLASSES)))
    assert all(int(q[i].argmax()) == i for i in range(len(CLASSES)))


def test_soft_targets_give_more_mass_to_nearer_classes():
    loss = AngularSoftTargetLoss(angular_cost_matrix(CLASSES, ANGLE_MAP), tau=45.0)
    i = CLASSES.index("regular_facing_right")
    near = loss.soft_targets[i, CLASSES.index("diag_up_right_facing_down")]
    far = loss.soft_targets[i, CLASSES.index("regular_facing_left")]
    assert near > far


def test_small_tau_recovers_plain_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, len(CLASSES))
    target = torch.randint(0, len(CLASSES), (8,))
    angular = AngularSoftTargetLoss(angular_cost_matrix(CLASSES, ANGLE_MAP), tau=1e-3)
    assert angular(logits, target).item() == pytest.approx(
        nn.CrossEntropyLoss()(logits, target).item(), abs=1e-4)


def test_class_weights_are_honoured_like_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, len(CLASSES))
    target = torch.randint(0, len(CLASSES), (8,))
    w = torch.tensor([2.0, 0.5, 1.0, 3.0])
    angular = AngularSoftTargetLoss(angular_cost_matrix(CLASSES, ANGLE_MAP), tau=1e-3,
                                    weight=w)
    expected = nn.CrossEntropyLoss(weight=w, reduction="none")(logits, target)
    # CrossEntropyLoss's weighted mean divides by the summed weights.
    assert angular(logits, target).item() == pytest.approx(
        (expected.sum() / w[target].sum()).item(), abs=1e-4)


def test_a_near_miss_costs_less_than_a_far_miss():
    """The whole point: being 45° wrong must hurt less than being 180° wrong."""
    loss = AngularSoftTargetLoss(angular_cost_matrix(CLASSES, ANGLE_MAP), tau=45.0)
    truth = torch.tensor([CLASSES.index("regular_facing_right")])

    def confident_in(name):
        logits = torch.full((1, len(CLASSES)), -5.0)
        logits[0, CLASSES.index(name)] = 5.0
        return loss(logits, truth).item()

    assert confident_in("regular_facing_right") < confident_in("diag_up_right_facing_down")
    assert confident_in("diag_up_right_facing_down") < confident_in("regular_facing_left")


def test_plain_cross_entropy_cannot_tell_the_two_misses_apart():
    """Contrast with the baseline: CE charges the same for both mistakes."""
    ce = nn.CrossEntropyLoss()
    truth = torch.tensor([CLASSES.index("regular_facing_right")])

    def confident_in(name):
        logits = torch.full((1, len(CLASSES)), -5.0)
        logits[0, CLASSES.index(name)] = 5.0
        return ce(logits, truth).item()

    assert confident_in("diag_up_right_facing_down") == pytest.approx(
        confident_in("regular_facing_left"))


def test_gradient_flows():
    loss = AngularSoftTargetLoss(angular_cost_matrix(CLASSES, ANGLE_MAP), tau=45.0)
    logits = torch.randn(4, len(CLASSES), requires_grad=True)
    loss(logits, torch.zeros(4, dtype=torch.long)).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

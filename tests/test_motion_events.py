import numpy as np
from g05.data.motion_events import classify_motion_windows


def source(yaw):
    a, s = np.zeros((len(yaw), 23)), np.zeros((len(yaw), 61))
    a[:, 2] = yaw
    a[:, [14, 22]] = 1
    s[:, 2] = np.r_[yaw[0], yaw[:-1]]
    return a, s


def test_turn_then_stop_and_start_reverse_are_not_plain_holds():
    a, s = source([.2]*20 + [0]*20 + [-.2]*20 + [.2]*20)
    c = classify_motion_windows(a, s)
    assert c[0] == 0
    assert c[10] == 1
    assert c[20] == 1  # first stop action while physical yaw still nonzero
    assert c[30] == 2
    assert c[50] == 2


def test_mean_zero_alternation_is_not_a_stop():
    a, s = source([.2, -.2]*24)
    assert not classify_motion_windows(a, s).any()


def test_full_stable_suffix_required_inside_executed_window():
    a, s = source([.2]*15 + [0]*20)
    c = classify_motion_windows(a, s)
    assert c[0] == 0  # only one stopped action, rest outside first16
    assert c[3] == 1  # four stopped actions inside first16


def test_gripper_boundary_and_partial_end_not_dropped():
    a, s = source([0]*40)
    a[10:, 14] = -1
    c = classify_motion_windows(a, s)
    assert c[0] == 3 and c[10] == 3 and c[11] == 0
    assert c.shape == (40,) and not c[-15:].any()

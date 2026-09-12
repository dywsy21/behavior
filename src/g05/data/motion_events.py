"""Full-window base transition labels from raw expert actions (no averaging)."""
import numpy as np


def classify_motion_windows(actions, states, horizon=16, stable=4):
    actions, states = np.asarray(actions), np.asarray(states)
    if actions.ndim != 2 or states.shape != (len(actions), 61) or actions.shape[1] != 23:
        raise ValueError("Expected aligned raw R1Pro action[time,23]/state[time,61]")
    if horizon < stable or len(actions) < horizon or not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Invalid motion window inputs")
    yaw = actions[:, 2]
    n = len(yaw)
    previous = np.r_[yaw[0], yaw[:-1]]
    # Inspect consecutive future targets, not their mean: alternating positive
    # and negative yaw is not a stopped robot.
    chunks = np.lib.stride_tricks.sliding_window_view(yaw, stable)
    stopped = np.all(np.abs(chunks) <= .01, axis=1)
    positive = np.all(chunks > .05, axis=1)
    negative = np.all(chunks < -.05, axis=1)
    yaw_stop = np.zeros(n, bool)
    yaw_start_reverse = np.zeros(n, bool)
    length = len(chunks)
    yaw_stop[:length] = stopped & ((np.abs(previous[:length]) > .05) | (np.abs(states[:length, 2]) > .05))
    yaw_start_reverse[:length] = ((positive & (previous[:length] <= .01)) |
                                 (negative & (previous[:length] >= -.01)))
    grip = np.zeros(n, bool)
    grip[1:] = np.any(actions[1:, [14, 22]] != actions[:-1, [14, 22]], axis=1)
    size = n - horizon + 1
    events = {}
    # A stable suffix must fit inside the actual execution horizon.
    for name, signal in (("yaw_stop", yaw_stop), ("yaw_start_reverse", yaw_start_reverse)):
        events[name] = np.lib.stride_tricks.sliding_window_view(signal, horizon - stable + 1)[:size].any(axis=1)
    events["gripper"] = np.lib.stride_tricks.sliding_window_view(grip, horizon).any(axis=1)
    category = np.zeros(n, dtype=np.uint8)
    category[:size] = np.where(events["yaw_stop"], 1,
        np.where(events["yaw_start_reverse"], 2, np.where(events["gripper"], 3, 0)))
    return category

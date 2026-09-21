"""Fixed, bidirectional subpixel localization of existing SIFT matches.

This is an explicit experimental frontend, not a motion solver or fallback.
It cannot accept a correspondence on the basis of a fitted scene transform.
"""
import numpy as np


VERSION = "sift_initialized_bidirectional_lk_v1"


def refine_matches(before, after, old_uv, proposed_uv):
    import cv2

    before, after = np.asarray(before), np.asarray(after)
    if (before.ndim != 2 or before.shape != after.shape or min(before.shape) < 8
            or before.dtype != np.uint8 or after.dtype != np.uint8):
        raise ValueError("Aligned current grayscale uint8 images required")
    old, proposed = np.asarray(old_uv, dtype=float), np.asarray(proposed_uv, dtype=float)
    if (old.ndim != 2 or old.shape[1:] != (2,) or proposed.shape != old.shape
            or not np.isfinite(old).all() or not np.isfinite(proposed).all()):
        raise ValueError("Finite aligned pixel coordinates required")
    height, width = before.shape

    def inside(uv):
        # A full depth patch must remain available after tracking as well.
        rounded = np.rint(uv)
        return (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 1) & (uv[:, 0] < width-1)
                & (uv[:, 1] >= 1) & (uv[:, 1] < height-1)
                & (rounded[:, 0] >= 1) & (rounded[:, 0] <= width-2)
                & (rounded[:, 1] >= 1) & (rounded[:, 1] <= height-2))

    output = proposed.copy()
    accepted = np.zeros(len(old), dtype=bool)
    candidates = np.flatnonzero(inside(old) & inside(proposed))
    receipt = {"version": VERSION, "input_matches": len(old), "in_bounds_candidates": len(candidates),
               "forward_supported": 0, "bidirectional_accepted": 0,
               "window": [21, 21], "pyramid_levels": 3, "max_iterations": 30,
               "epsilon_px": .01, "forward_backward_max_px": 1.,
               "fit_residuals_not_used": True, "unrefined_fallback": False}
    if not len(candidates):
        return output, accepted, receipt
    first = old[candidates].astype(np.float32).reshape(-1, 1, 2)
    guess = proposed[candidates].astype(np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    try:
        forward, status, _ = cv2.calcOpticalFlowPyrLK(before, after, first, guess, **options)
        if forward is None or status is None:
            return output, accepted, {**receipt, "reason": "NO_FORWARD_TRACKS"}
        forward = forward.reshape(-1, 2)
        if forward.shape != (len(candidates), 2) or status.size != len(candidates):
            raise ValueError("Invalid tracker result shape")
        good = (status.reshape(-1) == 1) & inside(forward)
        selected = np.flatnonzero(good)
        receipt["forward_supported"] = len(selected)
        if not len(selected):
            return output, accepted, {**receipt, "reason": "NO_FORWARD_TRACKS"}
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            after, before, forward[selected].astype(np.float32).reshape(-1, 1, 2),
            first[selected].copy(), **options)
        if backward is None or reverse_status is None:
            return output, accepted, {**receipt, "reason": "NO_REVERSE_TRACKS"}
        backward = backward.reshape(-1, 2)
        if backward.shape != (len(selected), 2) or reverse_status.size != len(selected):
            raise ValueError("Invalid reverse tracker result shape")
        error = np.linalg.norm(backward-first[selected, 0], axis=1)
        keep = (reverse_status.reshape(-1) == 1) & inside(backward) & (error <= 1.)
        indices = candidates[selected[keep]]
        output[indices] = forward[selected[keep]]
        accepted[indices] = True
        receipt["bidirectional_accepted"] = int(accepted.sum())
        receipt["median_forward_backward_error_px"] = float(np.median(error[keep])) if keep.any() else None
    except cv2.error:
        # Never silently restore the old matches after a failed tracking call.
        return proposed.copy(), np.zeros(len(old), dtype=bool), {**receipt, "bidirectional_accepted": 0,
                                                               "reason": "TRACKER_ERROR_NO_FALLBACK"}
    return output, accepted, receipt

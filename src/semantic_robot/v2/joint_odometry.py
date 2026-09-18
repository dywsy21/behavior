"""One fixed RGB-D estimator: robust 3D initialization + joint refinement.

This is never a per-frame solver fallback. Every supported frame pair follows
the same objective, including those whose unrefined 3D fit already passes. The
pixel/metre scales normalize existing engineering limits; they are not claimed
to be calibrated sensor uncertainties. Static-world support remains assumed.
"""
import numpy as np

from .odometry import body_motion, solve_rgbd_correspondences


def _camera(T):
    T = np.asarray(T, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-5)):
        raise ValueError("Finite proper robot-camera SE3 required")
    return T


def solve_joint_correspondences(points_before, pixels_after, points_after, K, camera_before, camera_after):
    import cv2
    from scipy.optimize import least_squares

    a, u1, b = (np.asarray(x, dtype=float) for x in (points_before, pixels_after, points_after))
    result = {"valid": False, "reason": "INSUFFICIENT_STABLE_CORRESPONDENCES",
              "matches": len(a), "estimator": "RGBD_FIXED_JOINT_REFINEMENT",
              "solver_fallback": False}
    if len(a) < 25:
        return result
    if (a.shape != (len(a), 3) or b.shape != a.shape or u1.shape != (len(a), 2)
            or not all(np.isfinite(x).all() for x in (a, b, u1))):
        raise ValueError("Finite aligned RGB-D correspondences required")
    K = np.asarray(K, dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-8)
            or abs(np.linalg.det(K)) < 1e-9):
        raise ValueError("Finite nondegenerate calibrated intrinsics required")
    before, after = _camera(camera_before), _camera(camera_after)
    if np.any(a[:, 2] <= .01) or np.any(b[:, 2] <= .01):
        return {**result, "reason": "INVALID_OBSERVED_DEPTH"}
    projection = a @ K.T
    u0 = projection[:, :2] / projection[:, 2, None]
    # SIFT may emit orientation/scale copies at the same image location; also
    # reject many-to-one pixel matches. Keep the first original deterministic
    # correspondence, before any fit/error-dependent selection.
    seen0, seen1, indices = set(), set(), []
    for index, (old, new) in enumerate(zip(np.rint(u0), np.rint(u1))):
        left, right = tuple(old), tuple(new)
        if left not in seen0 and right not in seen1:
            seen0.add(left); seen1.add(right); indices.append(index)
    a, b, u0, u1 = (x[indices] for x in (a, b, u0, u1))
    result["unique_matches"] = len(a)
    if len(a) < 25:
        return {**result, "reason": "INSUFFICIENT_UNIQUE_PIXEL_CORRESPONDENCES"}
    initializer = solve_rgbd_correspondences(a, u1, b, K, before, after)
    if "body_transform_current_in_previous" not in initializer:
        return {**result, "reason": "NO_ROBUST_3D_INITIALIZATION", "initializer_reason": initializer["reason"]}
    S = np.diag([1., -1., -1., 1.])
    rel = S @ np.linalg.inv(after) @ np.linalg.inv(np.asarray(initializer["body_transform_current_in_previous"])) @ before @ S
    support = np.linalg.norm(a @ rel[:3, :3].T + rel[:3, 3] - b, axis=1) < .01
    if support.sum() < 25 or support.mean() < .45:
        return {**result, "reason": "INSUFFICIENT_ROBUST_3D_SUPPORT"}
    aa, bb, uu0, uu1 = (x[support] for x in (a, b, u0, u1))
    x0 = np.r_[cv2.Rodrigues(rel[:3, :3])[0].ravel(), rel[:3, 3]]

    def errors(x):
        rotation = cv2.Rodrigues(x[:3])[0]; translation = x[3:]
        forward = aa @ rotation.T + translation
        backward = (bb - translation) @ rotation
        if (not np.isfinite(forward).all() or not np.isfinite(backward).all()
                or np.any(forward[:, 2] <= .01) or np.any(backward[:, 2] <= .01)):
            raise ValueError("Nonfinite/behind-camera refinement")
        pf, pb = forward @ K.T, backward @ K.T
        return pf[:, :2]/pf[:, 2, None]-uu1, pb[:, :2]/pb[:, 2, None]-uu0, forward-bb

    def objective(x):
        pixel, reverse, point = errors(x)
        return np.concatenate([pixel.ravel(), reverse.ravel(), (point/.01).ravel()])

    try:
        solution = least_squares(objective, x0, loss="soft_l1", f_scale=1., max_nfev=50)
        if not solution.success or not np.isfinite(solution.x).all():
            return {**result, "reason": "JOINT_OPTIMIZATION_NOT_CONVERGED"}
        pixel, reverse, point = (np.linalg.norm(x, axis=1) for x in errors(solution.x))
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return {**result, "reason": "JOINT_OPTIMIZATION_INVALID"}
    # Evaluate every frozen initializer-support point, not a new favorable
    # subset. Count the points still agreeing with both image directions and
    # 3D, without refitting or dropping the rest from median validation.
    agreement = (point < .01) & (pixel < 1.8) & (reverse < 1.8)
    result.update(initial_3d_support=int(support.sum()), inliers=int(agreement.sum()),
                  inlier_fraction=float(agreement.sum()/len(a)),
                  median_reprojection_px=float(np.median(pixel)),
                  median_reverse_reprojection_px=float(np.median(reverse)),
                  median_depth_correspondence_m=float(np.median(point)),
                  refinement_nfev=int(solution.nfev), fixed_support_no_reselection=True,
                  normalization={"pixel": 1., "point_m": .01})
    if result["inliers"] < 25 or result["inlier_fraction"] < .45:
        return {**result, "reason": "INSUFFICIENT_JOINT_RGBD_SUPPORT"}
    if np.median(pixel) > 1. or np.median(reverse) > 1. or np.median(point) > .015:
        return {**result, "reason": "RGB_DEPTH_MOTION_DISAGREEMENT"}
    rel = np.eye(4)
    rel[:3, :3] = cv2.Rodrigues(solution.x[:3])[0]; rel[:3, 3] = solution.x[3:]
    motion, delta = body_motion(before, after, rel)
    if (not np.isfinite(motion).all() or np.linalg.norm(delta[:2]) > .18
            or abs(delta[2]) > .30 or abs(motion[2, 3]) > .035):
        return {**result, "reason": "OUTSIDE_SINGLE_MICRO_ACTION_MOTION_BOUND"}
    return {**result, "valid": True, "reason": "CURRENT_RGBD_JOINT_MOTION",
            "body_delta": delta.tolist(), "body_translation_z_m": float(motion[2, 3]),
            "body_transform_current_in_previous": motion.tolist()}

"""Robot-only optimistic reach bounds: can VETO arrival, never prove it."""
import numpy as np


def navigation_workspace_check(model, q, target, arms):
    result={"valid":False,"within_optimistic_reach":False,"per_arm":{},
            "source":"onboard_target_depth_and_robot_link_lengths",
            "not_a_reachable_pose_or_arrival_certificate":True,
            "reason":"TARGET_OR_ROBOT_GEOMETRY_UNKNOWN"}
    if not target.get("valid"):return result
    point=np.asarray(target["point_base_m"],dtype=float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError("Finite target point required")
    chains=model.spec.get("metadata",{}).get("arm_chains",{})
    centers=model.grasp_centers(q)
    for arm in arms:
        chain=chains.get(arm,[])
        if len(chain)<2 or any(name not in model.links for name in chain):return result
        positions=[model.forward(q,name)[:3,3] for name in chain]+[centers[arm]]
        # Triangle inequality: even a perfectly straight arm cannot exceed
        # total link length. Horizontal projection is deliberately optimistic;
        # no IK, collision-free pose or target identity is certified by it.
        radius=float(sum(np.linalg.norm(b-a) for a,b in zip(positions,positions[1:])))
        if not .05<radius<3.:return result
        distance=float(np.linalg.norm(point[:2]-positions[0][:2]))
        result["per_arm"][arm]={"shoulder_horizontal_distance_m":distance,
            "sum_link_length_m":radius,"outside_reach_lower_bound_m":max(0.,distance-radius),
            "within_optimistic_reach":distance<=radius}
    result["valid"]=bool(result["per_arm"])
    result["within_optimistic_reach"]=bool(result["valid"] and any(r["within_optimistic_reach"] for r in result["per_arm"].values()))
    result["reason"]="NOT_EXCLUDED_BY_REACH_BOUND" if result["within_optimistic_reach"] else "DESTINATION_BEYOND_ALL_ACTIVE_ARM_REACH_BOUNDS"
    return result

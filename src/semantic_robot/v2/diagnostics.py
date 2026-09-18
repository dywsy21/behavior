"""Privileged post-action audit ONLY; never included in policy observations."""


def grasp_audit(robot):
    mode=robot.grasping_mode
    result={"source":"simulator_diagnostic_only","not_available_to_actor":True,
            "grasping_mode":mode,"assisted_objects":None,
            "not_a_task_success_or_physical_grip_strength_test":True}
    if mode in ("assisted","sticky"):
        result["assisted_objects"]={arm:None if obj is None else obj.name
                                    for arm,obj in robot._ag_obj_in_hand.items()}
    return result

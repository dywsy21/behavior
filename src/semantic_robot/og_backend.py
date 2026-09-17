"""Adapter to OG v3.9 R1Pro kinematics, not a scene-truth policy.

This private in-simulator development backend accesses ONLY robot joint state,
robot-relative link poses and robot Jacobians (same quantities as native IK).
Formal remote-policy submission needs a FK/Jacobian implementation from the
robot model and transmitted proprioception instead of this simulator adapter.
It must never transmit robot global pose, scene objects, contacts or predicates.
"""
import numpy as np

from .control import RobotState


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class OGKinematics:
    def __init__(self, robot):
        from omnigibson.utils.usd_utils import ControllableObjectViewAPI
        self.api = ControllableObjectViewAPI
        self.robot = robot
        self.path = robot.articulation_root_path
        expected = {"base": list(range(3)), "trunk": list(range(3, 7)),
                    "arm_left": list(range(7, 14)), "gripper_left": [14],
                    "arm_right": list(range(15, 22)), "gripper_right": [22]}
        actual = {k: array(v).tolist() for k, v in robot.controller_action_idx.items() if len(v)}
        if actual != expected or robot.action_dim != 23:
            raise ValueError(f"Unsupported controller layout; refuse guessed padding: {actual}")
        self.indices = np.r_[array(robot.trunk_control_idx), array(robot.arm_control_idx["left"]),
                             array(robot.arm_control_idx["right"])].astype(int)
        if len(self.indices) != 18 or len(set(self.indices)) != 18:
            raise ValueError("Expected four trunk + two seven-DOF arms")
        self.links = {"left": robot.eef_link_names["left"], "right": robot.eef_link_names["right"],
                      "torso": robot.joints[robot.trunk_joint_names[-1]].body1.split("/")[-1]}
        self.lower = array(robot.joint_lower_limits)[self.indices]
        self.upper = array(robot.joint_upper_limits)[self.indices]

    def state(self):
        joint = array(self.robot.get_joint_positions())
        jac = array(self.api.get_all_relative_jacobians(self.path))
        if jac.shape[0] != 1:
            raise ValueError("This development adapter requires one robot per view")
        offset = jac.shape[-1] - len(joint)
        if offset not in (0, 6):
            raise ValueError("Unexpected floating-base Jacobian layout")
        poses, jacobians = {}, {}
        for name, link in self.links.items():
            poses[name] = tuple(array(v) for v in self.api.get_link_relative_position_orientation(self.path, link))
            row = self.api.get_link_index(self.path, link) - 1
            jacobians[name] = jac[0, row][:, self.indices + offset]
        # Native proprio getter computes local base velocities; whitelist selected fields.
        proprio = self.robot._get_proprioception_dict()
        return RobotState(joint[self.indices], self.lower, self.upper, poses, jacobians,
                          np.array([array(proprio[f"gripper_{arm}_qpos"]).mean() for arm in ("left", "right")]),
                          array(proprio["base_qvel"]))

    def receipt(self):
        return {"native_action_dim": 23, "joint_indices": self.indices.tolist(), "links": self.links,
                "frame": "robot_base", "quaternion": "xyzw", "kinematics_source": "OG_native_robot_only",
                "global_pose_sent": False, "scene_truth_sent": False, "formal_remote_fk_port_pending": True}

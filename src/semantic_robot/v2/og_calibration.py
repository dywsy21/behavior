"""Development-only export of robot geometry; the deployed FK has no OG dependency."""
import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.og_backend import OGKinematics, array
from .kinematics import RobotModel, transform


class CalibratedRobot(OGKinematics):
    def calibrate(self):
        import omnigibson.lazy as lazy
        state = self.state()
        poses, jacobians = dict(state.poses), dict(state.jacobians)
        all_jac = array(self.api.get_all_relative_jacobians(self.path))[0]
        joint_count = len(array(self.robot.get_joint_positions()))
        offset = all_jac.shape[-1]-joint_count
        for name in self.robot.links:
            try:
                row = self.api.get_link_index(self.path, name)-1
                if row < 0:
                    continue
                poses["link:"+name] = tuple(array(v) for v in self.api.get_link_relative_position_orientation(self.path, name))
                jacobians["link:"+name] = all_jac[row][:, self.indices+offset]
            except (KeyError, ValueError, AssertionError):
                continue  # non-articulation decorative links are not collision/FK links
        cameras = {}
        sensor_names = {"head": "zed_link", "left_wrist": "left_realsense_link", "right_wrist": "right_realsense_link"}
        for view, parent_name in sensor_names.items():
            sensors = [s for name, s in self.robot.sensors.items() if parent_name in name and hasattr(s, "intrinsic_matrix")]
            if len(sensors) != 1:
                raise ValueError(f"Ambiguous/missing {view} sensor: {len(sensors)}")
            sensor = sensors[0]
            prim, local = sensor.prim, np.eye(4)
            parent_path = str(self.robot.links[parent_name].prim.GetPath())
            for _ in range(8):
                if str(prim.GetPath()) == parent_path:
                    break
                local = np.asarray(lazy.pxr.UsdGeom.Xformable(prim).GetLocalTransformation()).T @ local
                prim = prim.GetParent()
            else:
                raise ValueError("Sensor not attached to expected robot link")
            p, q = poses["link:"+parent_name]
            T = transform(p, q) @ local
            if not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-4):
                raise ValueError("Camera calibration contains unhandled scale")
            J = jacobians["link:"+parent_name].copy()
            J[:3] += np.cross(J[3:].T, T[:3, 3]-p).T
            name = "camera_"+view
            poses[name] = (T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
            jacobians[name] = J
            cameras[view] = {"K": array(sensor.intrinsic_matrix).tolist(),
                             "width": int(sensor.image_width), "height": int(sensor.image_height),
                             "frame": "usd_camera_x_right_y_up_minus_z_forward",
                             "parent_link": parent_name, "T_parent_camera": local.tolist()}
        joint_names = list(self.robot.joints)
        chains = {}
        for arm, indices in (("left", self.indices[4:11]), ("right", self.indices[11:18])):
            names = ["link:"+self.robot.joints[joint_names[int(i)]].body1.split("/")[-1] for i in indices]
            if any(name not in poses for name in names):
                raise ValueError("Missing arm collision chain")
            chains[arm] = names
        metadata = {"cameras": cameras, "arm_chains": chains, "joint_indices": self.indices.tolist(),
                    "joint_names": [joint_names[int(i)] for i in self.indices],
                    "source": "robot_only_reference_poses_and_jacobians", "scene_truth": False,
                    "collision": "3cm arm capsules, 8cm wrist separation; not environment mesh collision"}
        return RobotModel.from_reference(state.q, state.lower, state.upper, poses, jacobians, metadata)

    def compare(self, model):
        state = self.state()
        predicted = model.state(state.q, state.gripper, state.base_velocity)
        result = {}
        for name in ("left", "right", "torso"):
            p, q = predicted.poses[name]; p0, q0 = state.poses[name]
            result[name] = {"position_m": float(np.linalg.norm(p-p0)),
                            "angle_rad": float(np.linalg.norm((Rotation.from_quat(q)*Rotation.from_quat(q0).inv()).as_rotvec())),
                            "jacobian_max_abs": float(np.max(np.abs(state.jacobians[name]-predicted.jacobians[name])))}
        for view, camera in model.spec["metadata"]["cameras"].items():
            p, q = (array(v) for v in self.api.get_link_relative_position_orientation(self.path, camera["parent_link"]))
            actual = transform(p, q) @ np.asarray(camera["T_parent_camera"])
            predicted_camera, _ = model.evaluate(state.q, "camera_"+view)
            result["camera_"+view] = {
                "position_m": float(np.linalg.norm(actual[:3,3]-predicted_camera[:3,3])),
                "angle_rad": float(np.linalg.norm(Rotation.from_matrix(actual[:3,:3]@predicted_camera[:3,:3].T).as_rotvec()))}
        return result

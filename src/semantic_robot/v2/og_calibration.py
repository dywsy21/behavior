"""Development-only export of robot geometry; the deployed FK has no OG dependency."""
import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.og_backend import OGKinematics, array
from .kinematics import RobotModel, transform, link_origin_jacobian


class CalibratedRobot(OGKinematics):
    def native_self_boxes(self):
        """Robot visual-link bounds at ACTUAL joint positions, including fingers.

        Robot asset meshes only; no scene mesh, segmentation or object queries.
        Conservative boxes may reject useful target points, never grant a grasp.
        """
        if not hasattr(self,"_self_bounds"):
            import omnigibson.lazy as lazy
            from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh
            self._self_bounds={}
            for name,link in self.robot.links.items():
                vertices=[]
                for mesh in link.visual_meshes.values():
                    prim,local=mesh.prim,np.eye(4)
                    for _ in range(16):
                        if str(prim.GetPath())==str(link.prim.GetPath()):break
                        local=np.asarray(lazy.pxr.UsdGeom.Xformable(prim).GetLocalTransformation()).T@local
                        prim=prim.GetParent()
                    else:raise ValueError("Robot visual mesh not attached to its declared link")
                    points=(array(mesh.points) if mesh.prim.GetPrimTypeInfo().GetTypeName()=="Mesh"
                            else np.asarray(mesh_prim_shape_to_trimesh_mesh(mesh.prim).vertices))
                    vertices.append((np.c_[points,np.ones(len(points))]@local.T)[:,:3])
                if vertices:
                    points=np.concatenate(vertices)
                    self._self_bounds[name]=(points.min(axis=0),points.max(axis=0))
        boxes=[]
        for name,(lower,upper) in self._self_bounds.items():
            # Decorative non-articulation links still have a robot-relative
            # transform through the link API; absence fails closed for verifier.
            p,q=(array(v) for v in self.api.get_link_relative_position_orientation(self.path,name))
            boxes.append({"link":name,"lower":lower.tolist(),"upper":upper.tolist(),
                          "T_base_link":transform(p,q).tolist()})
        return {"source":"robot_visual_link_boxes_actual_joint_fk","scene_truth":False,
                "includes_actual_finger_positions":True,"margin_m":.003,"boxes":boxes}

    def base_visual_surface(self):
        """Export the declared chassis skin, never any environment geometry."""
        import omnigibson.lazy as lazy
        from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh
        name=self.robot.base_footprint_link_name
        link=self.robot.links[name]
        vertices,faces=[],[]
        offset=0
        for mesh in link.visual_meshes.values():
            prim,local=mesh.prim,np.eye(4)
            for _ in range(16):
                if str(prim.GetPath())==str(link.prim.GetPath()):break
                local=np.asarray(lazy.pxr.UsdGeom.Xformable(prim).GetLocalTransformation()).T@local
                prim=prim.GetParent()
            else:raise ValueError("Base visual mesh not attached to declared chassis")
            if mesh.prim.GetPrimTypeInfo().GetTypeName()=="Mesh":
                points=array(mesh.points);indices=array(mesh.faces).astype(np.int64)
            else:
                primitive=mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                points=np.asarray(primitive.vertices);indices=np.asarray(primitive.faces)
            points=(np.c_[points,np.ones(len(points))]@local.T)[:,:3]
            vertices.append(points);faces.append(indices+offset);offset+=len(points)
        if not vertices:raise ValueError("Declared chassis has no visual geometry for self-depth filtering")
        return {"link":"link:"+name,"vertices":np.concatenate(vertices).round(8).tolist(),
                "faces":np.concatenate(faces).tolist(),"source":"robot_base_visual_mesh_only","scene_truth":False,
                "frame":"declared_base_link_local","not_environment_mesh":True}

    def native_grasp_regions(self):
        """Robot-defined finger contact strips, with actual finger q.

        Only robot asset points and robot-relative link transforms are read.
        No raycast, assisted-grasp status, object state or world pose is used.
        The two sides are equally weighted even when their point counts differ.
        """
        result = {}
        for arm in ("left", "right"):
            sides = []
            for definitions in (self.robot.assisted_grasp_start_points,
                                self.robot.assisted_grasp_end_points):
                points = []
                for entry in definitions[arm]:
                    p, q = (array(v) for v in self.api.get_link_relative_position_orientation(self.path, entry.link_name))
                    points.append((transform(p,q) @ np.r_[array(entry.position),1.])[:3])
                if not points:
                    raise ValueError("Robot grasp-region definition missing")
                sides.append(np.asarray(points))
            result[arm] = sides
        return result

    def native_grasp_centers(self):
        return {arm:np.mean([side.mean(axis=0) for side in sides],axis=0)
                for arm,sides in self.native_grasp_regions().items()}

    def local_com(self, name):
        if not hasattr(self, "_local_com"):
            self._local_com = {}
        if name not in self._local_com:
            self._local_com[name] = array(self.robot.links[name].center_of_mass).reshape(3).copy()
        return self._local_com[name]

    def state(self):
        state = super().state()
        for name, link in self.links.items():
            state.jacobians[name] = link_origin_jacobian(
                state.jacobians[name], state.poses[name][1], self.local_com(link))
        return state

    def calibrate(self, grounded=False):
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
                jacobians["link:"+name] = link_origin_jacobian(
                    all_jac[row][:, self.indices+offset], poses["link:"+name][1], self.local_com(name))
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
                    "source": "robot_only_reference_poses_and_com_corrected_jacobians", "scene_truth": False,
                    "jacobian_point": "link_origin_corrected_from_physx_com",
                    "local_link_com": {name: value.tolist() for name, value in self._local_com.items()},
                    "collision": "3cm arm capsules, 8cm wrist separation; not environment mesh collision"}
        if grounded:
            metadata["base_visual_surface"]=self.base_visual_surface()
            native_centers = self.native_grasp_centers()
            metadata["grasp_centers_eef"] = {
                arm:(np.linalg.inv(transform(*state.poses[arm])) @ np.r_[point,1.])[:3].tolist()
                for arm,point in native_centers.items()}
            metadata["grasp_center_source"] = "equal_side_average_of_robot_finger_region_asset_points"
            metadata["grasp_center_invariance_requires_open_close_gate"] = True
            metadata["grasp_region_reference_gripper_m"]=state.gripper.tolist()
            all_q=array(self.robot.get_joint_positions());all_upper=array(self.robot.joint_upper_limits)
            metadata["grasp_region_reference_fully_open"]={arm:bool(all(
                abs(all_q[joint_names.index(name)]-all_upper[joint_names.index(name)])<.0001
                for name in self.robot.finger_joint_names[arm])) for arm in ("left","right")}
            metadata["grasp_regions_eef"]={arm:[
                (np.c_[side,np.ones(len(side))] @ np.linalg.inv(transform(*state.poses[arm])).T)[:,:3].tolist()
                for side in sides] for arm,sides in self.native_grasp_regions().items()}
            metadata["grasp_region_source"]="robot_finger_contact_definitions_not_object_grasp_prediction"
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
        if "grasp_centers_eef" in model.spec["metadata"]:
            centers = model.grasp_centers(state.q)
            for arm, actual in self.native_grasp_centers().items():
                result["grasp_center_"+arm] = {"position_m":float(np.linalg.norm(actual-centers[arm])),
                                             "angle_rad":0.}
        if "grasp_regions_eef" in model.spec["metadata"]:
            actual=self.native_grasp_regions()
            for arm,row in model.grasp_regions(state.q,state.gripper).items():
                if row["valid"]:
                    error=max(float(np.linalg.norm(np.asarray(pred)-real,axis=1).max())
                              for pred,real in zip(row["sides_base_m"],actual[arm]))
                    result["grasp_region_"+arm]={"position_m":error,"angle_rad":0.}
        return result

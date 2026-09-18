"""Read-only privileged teacher measurements; never a deployed actor dependency.

No reset, teleport, state setter, simulator step or success override occurs here.
Installed API uncertainty raises/returns UNKNOWN, not a fabricated measurement.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from native_teacher_outcomes import rigid


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def pose(obj):
    p, q = map(array, obj.get_position_orientation())
    out = np.eye(4); out[:3, :3] = Rotation.from_quat(q).as_matrix(); out[:3, 3] = p
    return rigid(out)


def state_of(obj, name):
    matches = [value for key, value in obj.states.items() if key.__name__ == name]
    if len(matches) != 1: raise ValueError("Missing/ambiguous privileged state: "+name)
    return matches[0]


def truth(value):
    # Explicit enum names; UNKNOWN=-1 must NEVER become bool(-1)==True.
    name = getattr(value, "name", None)
    if name == "TRUE": return True
    if name == "FALSE": return False
    return None


def measured_bool(value):
    value = array(value)
    if value.shape != () or value.dtype.kind != "b":
        return None
    return bool(value)


class PrivilegedReader:
    def __init__(self, env, spec):
        from omnigibson.utils.usd_utils import RigidContactAPI
        from replay_contact_audit import audit_pairs
        self.api, self.pairs = RigidContactAPI, audit_pairs
        self.robot, self.spec, self.idx = env.robots[0], spec, env.scene.idx
        scope = env.task.object_scope
        self.objects = {}
        for name in [spec["target"], spec["destination"], *spec["payloads"]]:
            if not name: continue
            obj = scope.get(name)
            if obj is None or not hasattr(obj, "links") or not hasattr(obj, "states"):
                raise ValueError("Unresolved teacher identity: "+name)
            self.objects[name] = obj
        self.target = self.objects[spec["target"]]
        self.rows = set(self.api._PATH_TO_ROW_IDX[self.idx])
        self.cols = set(self.api._PATH_TO_COL_IDX[self.idx])
        self.queried = {self.robot, *self.objects.values()}
        self.links = {link.prim_path for obj in self.queried for link in obj.links.values()}
        self.robot_links = {link.prim_path for link in self.robot.links.values()}
        self.target_links = {link.prim_path for link in self.target.links.values()}
        self.fingers = {a: {link.prim_path for link in self.robot.finger_links[a]} for a in ("left", "right")}
        collision_links = {link.prim_path for obj in self.queried for link in obj.links.values()
                           if getattr(link, "collision_meshes", {})}
        if not collision_links or not collision_links <= self.rows|self.cols:
            raise ValueError("Incomplete physical contact registration; UNKNOWN")
        if any(not paths or not paths <= self.rows|self.cols for paths in self.fingers.values()):
            raise ValueError("Finger contact registration unavailable")
        self.initial_pairs = None
        self.baseline_receipt = None
        self.toggle_context = None
        self.toggle_observer = None
        if spec["verb"] == "PRESS":
            from native_teacher_toggle import ToggleObserver
            target_state = state_of(self.target, "ToggledOn")
            self.toggle_observer = ToggleObserver(target_state, self.marker_measurement)
            self.toggle_context = self.toggle_observer.installed()
            self.toggle_context.__enter__()

    def close(self):
        if self.toggle_context is not None:
            context, self.toggle_context = self.toggle_context, None
            context.__exit__(None, None, None)

    def marker_measurement(self):
        import omnigibson as og
        s = state_of(self.target, "ToggledOn")
        radius = float(np.min(array(s.visual_marker.extent*s.scale*s.link.scale)))
        position = array(s.visual_marker.get_position_orientation()[0])
        if not np.isfinite(radius) or radius <= 0 or position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid installed toggle marker sphere")
        hits = set()
        def callback(hit):
            hits.add(str(hit.rigid_body)); return True
        og.sim.psqi.overlap_sphere(radius=radius, pos=position.tolist(), reportFn=callback)
        all_fingers = {link.prim_path for robot in self.robot.scene.robots if robot.is_manipulation
                       for links in robot.finger_links.values() for link in links}
        current = {tuple(pair) for pair in self.pairs(self.api,self.idx,self.queried,
                                                     self.rows,self.links,self.cols,True)}
        return {"arm_marker_overlap": {a:bool(hits&paths) for a,paths in self.fingers.items()},
                "arm_target_contact": {a:any(set(pair)&paths and set(pair)&self.target_links for pair in current)
                                       for a,paths in self.fingers.items()},
                "other_robot_marker_overlap": bool((hits&all_fingers)-set.union(*self.fingers.values())),
                "marker_hits": sorted(hits&all_fingers), "marker_center":position.tolist(), "marker_radius":radius,
                "hand_poses":{a:pose(self.robot.eef_links[a]).tolist() for a in self.fingers},
                "goal_parent_pose":pose(s.link).tolist()}

    def goal(self):
        s = self.spec
        parent = (self.target if s["goal_frame"] == "target" else
                  state_of(self.target, "ToggledOn").link if s["goal_frame"] == "toggle_link" else
                  self.objects[s["destination"]])
        return pose(parent)@rigid(s["goal_pose_local"])

    def base(self):
        return pose(self.robot.base_footprint_link)

    def check_local_fk(self, state, frame):
        """Do not assume native EEF/link and calibrated body frames coincide."""
        base = self.base()
        for arm in ("left", "right"):
            p, q = state.poses[arm]
            local = np.eye(4); local[:3, 3] = p
            local[:3, :3] = Rotation.from_quat(q).as_matrix()
            expected, actual = base@local, rigid(frame["hand_poses"][arm])
            if (np.linalg.norm(expected[:3, 3]-actual[:3, 3]) > .003 or
                    np.linalg.norm(Rotation.from_matrix(expected[:3, :3].T@actual[:3, :3]).as_rotvec()) > .02):
                raise RuntimeError("Native teacher EEF/base and calibrated executor frames disagree")

    def read(self, tick):
        current = {tuple(pair) for pair in self.pairs(self.api, self.idx, self.queried,
                    self.rows, self.links, self.cols, True)}
        # Baseline is retained for review, not a safety certificate. New
        # robot contacts except target fingers fail this acquisition trajectory.
        if self.initial_pairs is None:
            self.initial_pairs = current
            self.baseline_receipt = sorted(current)
        permitted = {tuple(sorted((finger, link))) for arm in self.fingers.values()
                     for finger in arm for link in self.target_links}
        forbidden = [pair for pair in current-self.initial_pairs-permitted if set(pair)&self.robot_links]
        contact = {a: any(set(pair)&self.fingers[a] and set(pair)&self.target_links for pair in current)
                   for a in self.fingers}
        held = {a: truth(self.robot.is_grasping(a, self.target)) for a in self.fingers}
        payload = [measured_bool(state_of(self.objects[name], "OnTop").get_value(self.target))
                   for name in self.spec["payloads"]]
        payload_ok = None if any(v is None for v in payload) else all(payload)
        hand_poses = {a: pose(self.robot.eef_links[a]).tolist() for a in self.fingers}
        frame = {"tick": tick, "target_uid": self.spec["target"], "target_pose": pose(self.target).tolist(),
                 "hand_poses": hand_poses, "held": held, "finger_contact": contact,
                 "contacts_known": True, "payload_ok": payload_ok, "forbidden_contacts": forbidden,
                 "contact_pairs": sorted(current), "privileged_teacher_only": True}
        if self.spec["verb"] == "PRESS":
            frame["toggled"] = measured_bool(state_of(self.target, "ToggledOn").value)
            frame["goal_parent_pose"] = pose(state_of(self.target, "ToggledOn").link).tolist()
            frame["toggle_observer_active"] = self.toggle_observer.active
            frame["toggle_events"] = self.toggle_observer.drain()
        if self.spec["verb"].startswith("PLACE"):
            destination = self.objects[self.spec["destination"]]
            frame["goal_parent_pose"] = pose(destination).tolist()
            relation = "Inside" if self.spec["verb"] == "PLACE_IN" else "OnTop"
            frame["relation"] = measured_bool(state_of(self.target, relation).get_value(destination))
            dst_links = {link.prim_path for link in destination.links.values()}
            frame["supported"] = any(set(pair)&self.target_links and set(pair)&dst_links for pair in current)
            frame["linear_velocity"] = array(self.target.get_linear_velocity()).tolist()
            frame["angular_velocity"] = array(self.target.get_angular_velocity()).tolist()
            # The live calibrated opening is supplied by the collector, not
            # inferred from the last OPEN command.
            frame["finger_opening"] = {}
            if relation == "Inside":
                # Installed Inside is only an AABB-center test. Fail closed
                # until a reviewed all-corners volume adapter is supplied.
                frame["corners_inside"] = None
        return frame

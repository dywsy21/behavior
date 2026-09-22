import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from semantic_robot.v2.grounding import GroundedEvidence
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.skill_completion import (
    CaptureRef, ExecutedMotionRef, read_capture, prepare_window, verify_grasp_request)
from test_v2 import fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def evidence(**changes):
    value = dict(visible=True, view="right_wrist", target_uv=[.5, .5], enclosed=None,
                 co_moving=None, supported=None, effect=None, hazard="none", note="Visible surface.",
                 other_views=[], target_reference="unknown")
    value.update(changes)
    return GroundedEvidence.parse(json.dumps(value))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.model, initial = fixture()
        self.snapshot_id = 0
        self.geometry = {"source": "robot_visual_link_boxes_actual_joint_fk", "scene_truth": False,
            "includes_actual_finger_positions": True, "margin_m": .003,
            "boxes": [{"link": "finger", "lower": [-.01]*3, "upper": [.01]*3,
                       "T_base_link": np.eye(4).tolist()}]}
        self.model.spec["metadata"]["robot_visual_boxes_reference"] = copy.deepcopy(self.geometry)
        self.goal = Goal("pick", "bin", "right", "target follows lifted hand")
        self.records = []; q = initial.q.copy(); gripper = initial.gripper.copy(); clock = 10
        for i in range(3):
            directory = self.root / f"decision_{i:02d}"; directory.mkdir()
            before = self.capture(directory / "before", q, gripper, clock)
            token = "RIGHT_CLOSE" if i == 0 else "RIGHT_UP"
            status = "GRIPPER_COMMAND_COMPLETED" if i == 0 else "TARGET_REACHED"
            if i == 0: gripper[1] = .007
            else: q[13] += .01
            after = self.capture(directory / "after_settle", q, gripper, clock + 30)
            execution = {"token": token, "control_start": clock, "control_end": clock + 18,
                "feedback": {"status": status, "control_ticks": 18}}
            if i == 0:
                execution["feedback"].update(holding="UNKNOWN", official_success="NOT_AVAILABLE_TO_ACTOR",
                    pose_tracking_status="TARGET_REACHED", target_error_m={"left": 0., "right": 0.},
                    orientation_error_deg={"left": 0., "right": 0.})
                execution["feedback"]["gripper_execution"] = {
                    "version": "gripper-command-completion-v1", "part": "right", "move": "close",
                    "command_complete": True, "success_claim": False,
                    "planned_control_ticks": 18, "executed_control_ticks": 18,
                    "final_state_checks": {k: True for k in ("finite_proprio", "joint_bounds",
                        "robot_collision_free", "active_hand_within_divergence_envelope",
                        "inactive_hand_within_pose_tolerance")},
                    "limits": {"active_position_m": .018, "active_angle_deg": 9.,
                        "inactive_position_m": .0025, "inactive_angle_deg": 1.5}}
            path = directory / "execution.json"; dump(path, execution)
            self.records.append(ExecutedMotionRef(token, path, sha(path), before, after)); clock += 30
        self.current = self.records[-1].after
        self.locate = Mock(return_value=evidence())

    def capture(self, folder, q, gripper, control):
        self.snapshot_id += 1
        folder.mkdir(parents=True)
        state = self.model.state(q, gripper, np.zeros(3))
        images = {v: np.random.default_rng(control).integers(0, 256, (100,100,3), dtype=np.uint8)
                  for v in ("head", "left_wrist", "right_wrist")}
        depths = {v: np.ones((100,100), np.float32) for v in images}
        for v in images: Image.fromarray(images[v]).save(folder / (v + ".png"))
        np.savez(folder / "depth.npz", **depths)
        dh = {v: hashlib.sha256(d.tobytes()).hexdigest() for v,d in depths.items()}
        sensors = {v: {"depth_sha256": dh[v], "rgb_sha256": hashlib.sha256(images[v].tobytes()).hexdigest(),
            "same_sensor_current_render": True, "control_steps_in_capture": 0,
            "depth_units": "metres", "depth_convention": "distance_to_image_plane", "shape": [100,100],
            "snapshot_id": self.snapshot_id, "render_barrier_updates": 4, "modalities": ["rgb", "depth_linear"]} for v in images}
        dump(folder / "sensors.json", sensors); dump(folder / "robot_self_geometry.json", self.geometry)
        dump(folder / "proprio.json", {"eef_base_m": {a: state.poses[a][0].round(3).tolist() for a in ("left", "right")},
            "finger_opening_m": {a: round(float(gripper[i]),3) for i,a in enumerate(("left", "right"))},
            "base_velocity_local": [0.,0.,0.], "torso_joints_rad": np.asarray(q[:4]).round(3).tolist()})
        cap = {"clock": {"prefix_control": 100, "native_control": control}, "q": np.asarray(q).tolist(),
            "gripper": np.asarray(gripper).tolist(), "kinematic_model_sha256": self.model.sha,
            "source": "render_only_current_onboard_RGBD_and_robot_joint_FK", "scene_truth": False,
            "array_layout": {v: {"rgb_shape": [3,100,100], "depth_shape": [100,100], "depth_dtype": "float32",
                                   "rgb_bytes": 30000, "depth_bytes": 40000} for v in images}, "depth_array_sha256": dh,
            "files_sha256": {p.name: sha(p) for p in folder.iterdir()}}
        dump(folder / "capture.json", cap)
        return CaptureRef(folder, sha(folder / "capture.json"), cap["clock"])

    def modify_capture(self, ref, mutate):
        cap = json.loads((ref.directory / "capture.json").read_text()); mutate(cap)
        dump(ref.directory / "capture.json", cap)
        return replace(ref, sha256=sha(ref.directory / "capture.json"))

    def modify_execution(self, index, mutate):
        ref = self.records[index]; data = json.loads(ref.execution_path.read_text()); mutate(data)
        dump(ref.execution_path, data)
        self.records[index] = replace(ref, execution_sha256=sha(ref.execution_path))

    def verify(self, **kwargs):
        return verify_grasp_request(self.model, kwargs.pop("goal",self.goal), kwargs.pop("records",self.records),
            kwargs.pop("current",self.current), localize=self.locate,
            deadline=kwargs.pop("deadline",time.perf_counter()+60), **kwargs)

    def test_actual_capture_and_chain_bind_three_current_views(self):
        frame = read_capture(self.current, self.model)
        self.assertEqual(frame.control, 200); self.assertEqual(len(frame.images),3)
        frames, executions = prepare_window(self.model,self.goal,self.records,self.current,time.perf_counter()+60)
        self.assertEqual([f.control for f in frames],[140,170,200])
        self.assertEqual(len(executions),3)

    def test_bytes_clock_or_privileged_capture_not_admitted(self):
        cases = [lambda c:c.update(scene_truth=True), lambda c:c.update(object_pose=[0,0,0]),
                 lambda c:c.update(q=[0]*17), lambda c:c.update(kinematic_model_sha256="f"*64),
                 lambda c:c["clock"].update(native_control=True), lambda c:c.update(source="oracle")]
        original = (self.current.directory/"capture.json").read_bytes()
        for change in cases:
            (self.current.directory/"capture.json").write_bytes(original)
            bad = self.modify_capture(self.current,change)
            with self.assertRaises(ValueError): read_capture(bad,self.model)
        (self.current.directory/"capture.json").write_bytes(original)
        with self.assertRaises(ValueError):read_capture(replace(self.current,sha256="0"*64),self.model)
        (self.current.directory/"head.png").write_bytes(b"not the recorded PNG")
        with self.assertRaises(ValueError):read_capture(self.current,self.model)

    def test_failed_action_no_localization_and_no_controls(self):
        self.modify_execution(2,lambda d:d["feedback"].update(status="NO_MOTION_PROGRESS"))
        result=self.verify(); self.assertFalse(result["public_holding_verified"])
        self.assertEqual(result["physical_controls"],0);self.locate.assert_not_called()

    def test_sensor_unit_barrier_snapshot_and_layout_must_agree(self):
        ref=self.current; original=(ref.directory/"sensors.json").read_bytes()
        cap_original=(ref.directory/"capture.json").read_bytes()
        for key,value in (("snapshot_id",False),("snapshot_id",99),("render_barrier_updates",0),
                          ("control_steps_in_capture",True),("depth_units","millimetres"),
                          ("same_sensor_current_render",False)):
            sensors=json.loads(original);sensors["head"][key]=value;dump(ref.directory/"sensors.json",sensors)
            (ref.directory/"capture.json").write_bytes(cap_original)
            bad=self.modify_capture(ref,lambda c:c["files_sha256"].update({"sensors.json":sha(ref.directory/"sensors.json")}))
            with self.assertRaises(ValueError):read_capture(bad,self.model)
        (ref.directory/"sensors.json").write_bytes(original)
        (ref.directory/"capture.json").write_bytes(cap_original)
        bad=self.modify_capture(ref,lambda c:c["array_layout"]["head"].update(rgb_bytes=1))
        with self.assertRaises(ValueError):read_capture(bad,self.model)

    def test_request_must_follow_same_state_without_gaps(self):
        for current in (self.records[0].after, self.records[1].after):
            self.assertFalse(self.verify(current=current)["public_holding_verified"])
        self.modify_execution(1,lambda d:d.update(control_start=41))
        self.assertFalse(self.verify()["public_holding_verified"]);self.locate.assert_not_called()

    def test_release_wrong_hand_skipped_close_and_long_history_rejected(self):
        for records in (self.records[1:],self.records*2,list(reversed(self.records)),
                        [replace(self.records[0],token="RIGHT_OPEN"),*self.records[1:]],
                        [replace(self.records[1],before=None),*self.records[1:]]):
            self.assertFalse(self.verify(records=records)["public_holding_verified"])
        self.assertFalse(self.verify(goal=Goal("pick","bin","left","moves"))["public_holding_verified"])
        self.assertFalse(self.verify(goal=Goal("press","button","right","changes"))["public_holding_verified"])
        self.locate.assert_not_called()

    def test_success_claim_is_not_gripper_command_receipt(self):
        self.modify_execution(0,lambda d:d["feedback"]["gripper_execution"].update(success_claim=True))
        self.assertFalse(self.verify()["public_holding_verified"]);self.locate.assert_not_called()

    def test_close_requires_original_full_safe_execution_receipt(self):
        original = self.records[0]
        original_bytes = original.execution_path.read_bytes()
        cases = [lambda f: f["gripper_execution"].update(version="unknown"),
                 lambda f: f["gripper_execution"]["final_state_checks"].update(robot_collision_free=False),
                 lambda f: f["gripper_execution"].update(planned_control_ticks=17),
                 lambda f: f["gripper_execution"].update(executed_control_ticks=18.0),
                 lambda f: f["gripper_execution"].pop("limits"),
                 lambda f: f["gripper_execution"].pop("final_state_checks"),
                 lambda f: f.update(visual_gate_failure="failed"),
                 lambda f: f.update(holding="TRUE"),
                 lambda f: f["target_error_m"].update(right=.019),
                 lambda f: f["orientation_error_deg"].update(left=2.)]
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                original.execution_path.write_bytes(original_bytes)
                self.records[0] = original
                self.modify_execution(0, lambda e: mutate(e["feedback"]))
                self.assertEqual(self.verify()["status"], "UNKNOWN")
                self.locate.assert_not_called()
        original.execution_path.write_bytes(original_bytes)
        self.records[0] = original

    def test_completed_up_with_explicit_visual_failure_is_not_accepted(self):
        self.modify_execution(1, lambda e: e["feedback"].update(visual_gate_failure="CURRENT_DEPTH_REJECTED"))
        self.assertEqual(self.verify()["status"], "UNKNOWN")
        self.locate.assert_not_called()

    def test_missing_request_deadline_or_motion_status_no_calls(self):
        for status in ("CONTINUE","SUCCEEDED",True,None):
            self.assertFalse(self.verify(requested_status=status)["public_holding_verified"])
        for deadline in (time.perf_counter()-1,float("nan"),True):
            self.assertFalse(self.verify(deadline=deadline)["public_holding_verified"])
        self.locate.assert_not_called()

    def test_same_tick_render_uses_request_images_not_an_extra_lift(self):
        frame=read_capture(self.current,self.model)
        fresh=self.capture(self.root/"decision_03"/"before",frame.state.q,frame.state.gripper,100)
        frames,ex=prepare_window(self.model,self.goal,self.records,fresh,time.perf_counter()+60)
        self.assertEqual(len(frames),3); self.assertEqual(frames[-1].ref.directory,fresh.directory)
        stale=self.capture(self.root/"decision_04"/"before",frame.state.q+.001,frame.state.gripper,100)
        self.assertFalse(self.verify(current=stale)["public_holding_verified"])

    def mocked_measurements(self, rows):
        odometry=Mock();odometry.observe.return_value={"valid":True,"body_transform_current_in_previous":np.eye(4).tolist()}
        return (patch("semantic_robot.v2.skill_completion.RGBDMotion",return_value=odometry),
                patch("semantic_robot.v2.skill_completion.localize_target",return_value={"valid":True,"point_base_m":[.4,-.3,.8]}),
                patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=rows))

    def test_positive_holding_is_not_skill_or_official_success(self):
        row={"registered_pair_consistent":True,"hand_delta_m":[0,0,.01]}
        a,b,c=self.mocked_measurements([dict(row),dict(row)])
        with a,b,c:result=self.verify()
        self.assertTrue(result["public_holding_verified"]);self.assertTrue(result["not_skill_success"])
        self.assertTrue(result["not_official_task_success"]);self.assertEqual(result["localization_calls"],3)
        self.assertNotIn("success",result);self.assertEqual(result["physical_controls"],0)

    def test_bad_final_pair_cannot_cherry_pick_prior_motion(self):
        a,b,c=self.mocked_measurements([{"registered_pair_consistent":True,"hand_delta_m":[0,0,.02]},
                                        {"registered_pair_consistent":False}])
        with a,b,c:result=self.verify()
        self.assertFalse(result["public_holding_verified"])

    def test_model_holding_claims_or_unavailable_localization_abstain(self):
        for e in (evidence(enclosed=True), evidence(co_moving=True), evidence(supported=False),
                  evidence(target_reference="held_right"), evidence(hazard="collision")):
            self.locate.return_value=e
            self.assertFalse(self.verify()["public_holding_verified"])
        self.locate.side_effect=RuntimeError("service unavailable")
        self.assertFalse(self.verify()["public_holding_verified"])

    def test_deadline_after_localization_abstains_before_tracking(self):
        deadline=time.perf_counter()+60
        def late(*args):
            timer.return_value=deadline+1
            return evidence()
        self.locate.side_effect=late
        with patch("semantic_robot.v2.skill_completion.time.perf_counter",return_value=deadline-1) as timer:
            result=self.verify(deadline=deadline)
        self.assertFalse(result["public_holding_verified"]);self.assertEqual(result["localization_calls"],1)
        self.assertEqual(result["frames"],[])


if __name__ == "__main__":unittest.main()

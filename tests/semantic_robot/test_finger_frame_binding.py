import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from semantic_robot.v2.grounded_harness import GroundedController
from semantic_robot.v2.odometry import RGBDMotion
from semantic_robot.v2.self_odometry import FINGER_KEY, FINGER_VERSION, VERSION, make_frame, validate_frame
from semantic_robot.v2.substep_odometry import SubstepMotion
import test_self_odometry
from test_finger_kinematics import current, finger_spec
from test_substep_odometry import measured


RUNNER = Path(__file__).resolve().parents[2] / "scripts/semantic_robot/run_v2.py"


class FingerFrameBindingTests(unittest.TestCase):
    def setUp(self):
        self.case = test_self_odometry.SelfOdometryTests()
        self.case.setUp()
        self.model, self.state = self.case.model, self.case.state
        self.model.spec["metadata"]["finger_kinematics"] = finger_spec()
        self.state.finger_qpos = current((.02, .03), (.01, .04))
        self.state.gripper[:] = .025
        self.args = self.case.args

    def bound(self, control=0):
        return dict(robot_frame=self.case.frame(control), gripper=self.state.gripper,
                    control=control, finger_qpos=dict(self.state.finger_qpos))

    def test_saved_frame_binds_real_asymmetric_fingers_not_just_mean(self):
        bound = self.bound()
        frame = bound.pop("robot_frame"); bound.pop("gripper"); bound.pop("control")
        result = validate_frame(frame, *self.args, self.state.gripper, 0, **bound)
        self.assertEqual(frame["version"], FINGER_VERSION)
        self.assertEqual(len(result), 64)
        self.state.finger_qpos["left_j0"] += .001
        self.state.finger_qpos["left_j1"] -= .001
        self.assertEqual(frame[FINGER_KEY]["left_j0"], .02)
        self.assertAlmostEqual(np.mean([self.state.finger_qpos[n] for n in ("left_j0", "left_j1")]), .025)
        with self.assertRaisesRegex(ValueError, "individual finger"):
            validate_frame(frame, *self.args, self.state.gripper, 0, finger_qpos=self.state.finger_qpos)

    def test_omission_downgrade_and_invalid_record_never_fall_back(self):
        frame = self.case.frame()
        with self.assertRaises(ValueError):
            validate_frame(frame, *self.args, self.state.gripper, 0)
        edits = [lambda f: f.pop(FINGER_KEY), lambda f: f.update(version=VERSION),
                 lambda f: f.update({FINGER_KEY: None}), lambda f: f.update({FINGER_KEY: {}}),
                 lambda f: f[FINGER_KEY].update(left_j0=True),
                 lambda f: f[FINGER_KEY].update(left_j0=float("nan")),
                 lambda f: f[FINGER_KEY].update(unmeasured=.01)]
        for edit in edits:
            changed = copy.deepcopy(frame); edit(changed)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_frame(changed, *self.args, self.state.gripper, 0, finger_qpos=self.state.finger_qpos)

    def test_model_requires_named_positions_and_calibrated_names_limits_means(self):
        for positions in (None, {"unmeasured": .025},
                          current((-.01, .06), (.01, .04)), current((.02, .02), (.01, .04))):
            self.state.finger_qpos = positions
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                self.case.frame()

    def test_legacy_frame_only_works_with_legacy_model_and_state(self):
        old = test_self_odometry.SelfOdometryTests(); old.setUp()
        frame = old.frame()
        self.assertEqual(frame["version"], VERSION)
        self.assertEqual(set(frame), {"version", "control", "q", "gripper", "sensors", "geometry"})
        validate_frame(frame, *old.args, old.state.gripper, 0)
        old.model.spec["metadata"]["finger_kinematics"] = finger_spec()
        with self.assertRaises(ValueError): validate_frame(frame, *old.args, old.state.gripper, 0)
        with self.assertRaises(ValueError): old.frame()

    def test_named_state_without_robot_finger_calibration_is_not_certified(self):
        frame = self.case.frame()
        self.model.spec["metadata"].pop("finger_kinematics")
        for positions in (dict(self.state.finger_qpos), {"uncalibrated": 9.0}):
            self.state.finger_qpos = positions
            with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "requires robot finger calibration"):
                self.case.frame()
            with self.assertRaisesRegex(ValueError, "requires robot finger calibration"):
                validate_frame(frame, *self.args, self.state.gripper, 0, finger_qpos=positions)

    def test_actual_rgbd_estimator_rejects_stale_fingers_before_state_mutation(self):
        odom = RGBDMotion("rgbd_joint", exclude_robot=True)
        bound = self.bound()
        self.assertTrue(odom.observe(*self.args, **bound)["initial"])
        before = odom.previous
        wrong = {**bound, "finger_qpos": current((.021, .029), (.01, .04))}
        for kwargs in (wrong, {k: v for k, v in bound.items() if k != "finger_qpos"}):
            with self.assertRaises(ValueError): odom.observe(*self.args, **kwargs)
            self.assertIs(odom.previous, before)
        with self.assertRaises(ValueError):
            RGBDMotion("rgbd_joint").observe(*self.args, finger_qpos=self.state.finger_qpos)

    def test_substeps_bind_and_deliver_same_fingers_even_if_new_frame_is_consistent(self):
        base = Mock(); base.exclude_robot = True
        base.observe.side_effect = [{"valid": True, "initial": True}, measured()]
        motion = SubstepMotion(base)
        motion.observe(*self.args, **self.bound()); motion.begin(0)
        end = self.bound(6); sample = {k: v for k, v in end.items() if k != "control"}
        motion.sample(*self.args, 6, **sample)
        self.assertTrue(motion.finish(6)["valid"])
        original_positions = dict(self.state.finger_qpos)
        self.state.finger_qpos = current((.021, .029), (.01, .04))
        for kwargs in ({**end, "finger_qpos": self.state.finger_qpos}, self.bound(6)):
            with self.assertRaises(ValueError): motion.observe(*self.args, **kwargs)
            self.assertIsNotNone(motion.pending)
        self.state.finger_qpos = original_positions
        self.assertTrue(motion.observe(*self.args, **end)["valid"])
        self.assertEqual(base.observe.call_count, 2)
        for call in base.observe.call_args_list:
            self.assertEqual(call.kwargs["finger_qpos"], original_positions)

    def test_actual_controller_passes_fingers_to_real_estimator(self):
        c = SimpleNamespace(model=self.model, motion=RGBDMotion("rgbd_joint", exclude_robot=True),
                            odometry_self_exclusion=True, harness=SimpleNamespace(), pending_motion=None)
        frame = self.case.frame()
        result = GroundedController.update_motion(c, self.case.images, self.case.depths, self.state,
                                                 robot_frame=frame, control=0)
        self.assertTrue(result["initial"])
        self.state.finger_qpos = current((.021, .029), (.01, .04))
        with self.assertRaises(ValueError):
            GroundedController.update_motion(c, self.case.images, self.case.depths, self.state,
                                             robot_frame=frame, control=0)

    def test_actual_reanchor_propagates_all_three_named_frames_and_final_snapshot(self):
        import test_search_reanchor
        old = test_search_reanchor.SearchReanchorTests(); old.setup()
        old.model.spec["metadata"]["finger_kinematics"] = finger_spec()
        old.model.spec["metadata"]["robot_visual_boxes_reference"] = copy.deepcopy(self.case.geometry)
        old.state.finger_qpos = current()
        old.c.odometry_self_exclusion = True
        frames, factories = [], []
        def save(control, images, depths, sensor, state):
            frame = make_frame(images, depths, old.model, state, control, self.case.geometry)
            frames.append(frame); return frame
        def factory(kind, **kwargs):
            self.assertEqual(kwargs, {"exclude_robot": True})
            base = Mock(); base.exclude_robot = True
            base.observe.side_effect = [{"valid": True, "initial": True}, measured(), measured()]
            factories.append(base); return base
        with patch("semantic_robot.v2.search_reanchor.RGBDMotion", side_effect=factory):
            result, snapshot = old.c.search_recovery.attempt(
                old.c, old.failed, controls=36, action_limit=60, terminal=False, deadline=None,
                observe=lambda label: (old.images, old.depths, {}), state_now=lambda: old.state,
                issue_hold=lambda: False, save_snapshot=save)
        self.assertTrue(result["valid"]); self.assertEqual(len(snapshot), 8)
        self.assertEqual(snapshot[6], frames[-1]); self.assertEqual(snapshot[7], current())
        self.assertEqual([c.kwargs["control"] for c in factories[0].observe.call_args_list], [36, 42, 48])
        for call in factories[0].observe.call_args_list:
            self.assertEqual(call.kwargs["finger_qpos"], current())
        old.state.finger_qpos["left_j0"] = .02
        self.assertEqual(snapshot[7]["left_j0"], .05)

    def test_actual_runner_gate_substep_and_post_motion_forward_fingers(self):
        tree = ast.parse(RUNNER.read_text())
        for name, target in (("state", "gate_motion"), ("sample_state", "substep_motion"),
                             ("post_state", "gate_motion")):
            matches = []
            for parent in ast.walk(tree):
                body = getattr(parent, "body", None)
                if not isinstance(body, list): continue
                for i, node in enumerate(body[:-1]):
                    if (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "motion_kwargs"
                            and "gripper=" + name + ".gripper" in ast.unparse(node)):
                        matches.append(body[i:i+2])
            self.assertEqual(len(matches), 1)
            method = "sample" if name == "sample_state" else "observe"
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == target + "." + method
                     and any(ast.unparse(a) == name + ".q" for a in n.value.args)]
            self.assertEqual(len(calls), 1)
            code = compile(ast.fix_missing_locations(ast.Module(body=matches[0] + calls, type_ignores=[])),
                           str(RUNNER), "exec")
            for positions in (self.state.finger_qpos, None):
                state = copy.deepcopy(self.state); state.finger_qpos = positions
                estimator = Mock()
                ns = {"args": SimpleNamespace(odometry_self_exclusion=True), name: state, target: estimator,
                      "controls": 6, "model": self.model}
                for prefix in ("", "sample_", "post_"):
                    ns[prefix + "images"] = self.case.images
                    ns[prefix + "depths"] = self.case.depths
                ns.update(motion_frame={}, sample_frame={}, post_frame={})
                exec(code, ns)
                kwargs = getattr(estimator, method).call_args.kwargs
                if positions is None: self.assertNotIn("finger_qpos", kwargs)
                else: self.assertEqual(kwargs["finger_qpos"], positions)


class ImplementationDigestTests(unittest.TestCase):
    def setUp(self):
        node = next(n for n in ast.parse(RUNNER.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "implementation_digest")
        self.code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(RUNNER), "exec")
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = ("src/semantic_robot/control.py", "src/semantic_robot/og_backend.py",
                      "src/semantic_robot/v2/control.py", "src/semantic_robot/v2/nested/model.py",
                      "scripts/semantic_robot/run_v2.py", "scripts/semantic_robot/run_sim.py")
        import sys
        sys.path.insert(0, str(RUNNER.parent))
        from native_full_profile import DEPENDENCY_FILES
        self.files += tuple('scripts/semantic_robot/' + p for p in DEPENDENCY_FILES)

    def build(self, name):
        root = self.root / name
        for path in self.files:
            file = root / path; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(path)
        return root

    def get(self, root):
        ns = {"Path": Path, "REPO": root, "__file__": str(root / "scripts/semantic_robot/run_v2.py"),
              "hashlib": hashlib, "json": json}
        exec(self.code, ns); return ns["implementation_digest"]()

    def test_each_actual_proprio_runner_or_recursive_module_change_invalidates_gate(self):
        root = self.build("a"); expected = self.get(root)
        for path in self.files:
            file = root / path; file.write_text(path + "\nchanged")
            with self.subTest(path=path): self.assertNotEqual(self.get(root), expected)
            file.write_text(path)
        self.assertEqual(self.get(root), expected)

    def test_add_remove_and_rename_bind_module_identity_not_just_basename(self):
        root = self.build("a"); expected = self.get(root)
        extra = root / "src/semantic_robot/v2/extra.py"; extra.write_text("x")
        self.assertNotEqual(self.get(root), expected)
        extra.unlink(); self.assertEqual(self.get(root), expected)
        file = root / "src/semantic_robot/control.py"
        moved = file.with_name("renamed.py"); file.rename(moved)
        self.assertNotEqual(self.get(root), expected)
        moved.rename(file); file.unlink(); self.assertNotEqual(self.get(root), expected)
        (root / "scripts/semantic_robot/run_sim.py").unlink()
        with self.assertRaises(FileNotFoundError): self.get(root)

    def test_checkout_relocation_and_unrelated_docs_do_not_invalidate_gate(self):
        a, b = self.build("a"), self.build("different_parent/b")
        expected = self.get(a); self.assertEqual(expected, self.get(b))
        (a / "docs").mkdir(); (a / "docs/plan.md").write_text("unrelated")
        (a / "src/semantic_robot/v2/debug.log").write_text("unrelated")
        self.assertEqual(self.get(a), expected)


if __name__ == "__main__": unittest.main()

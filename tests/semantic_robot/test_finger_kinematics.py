import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.control import named_finger_positions
from semantic_robot.og_backend import OGKinematics
from semantic_robot.v2.finger_kinematics import FingerKinematics, SOURCE, link_reference
from semantic_robot.v2.kinematics import RobotModel, transform
from semantic_robot.v2.og_calibration import CalibratedRobot
from test_v2 import fixture


def finger_spec():
    spec = {"version": 1, "frame": "eef", "source": SOURCE, "scene_truth": False,
            "embodiment": "R1Pro_parallel_prismatic_jaws", "arms": {}}
    for arm in ("left", "right"):
        row = {"joint_names": [arm + "_j0", arm + "_j1"], "q_reference": [.05, .05],
               "lower": [0, 0], "upper": [.05, .05], "links": {}}
        for i, sign in enumerate((1, -1)):
            T = np.eye(4); T[:3, 3] = [0, sign * .05, .02]
            screws = np.zeros((6, 2)); screws[1, i] = sign
            row["links"][arm + "_f" + str(i)] = {
                "T_reference": T.tolist(), "screws": screws.tolist(),
                "grasp_strip_points_local_m": [[0, 0, -.01], [0, 0, .01]]}
        spec["arms"][arm] = row
    return spec


def current(left=(.05, .05), right=(.05, .05)):
    return {a + "_j" + str(i): v for a, pair in (("left", left), ("right", right))
            for i, v in enumerate(pair)}


def calibrated_fixture():
    model, state = fixture()
    spec = copy.deepcopy(model.spec)
    spec["metadata"]["finger_kinematics"] = finger_spec()
    model = RobotModel(spec)
    return model, model.state(state.q, state.gripper, state.base_velocity, current())


class FingerKinematicsTests(unittest.TestCase):
    def test_legacy_model_state_and_serialization_do_not_invent_fingers(self):
        model, state = fixture()
        self.assertIsNone(state.finger_qpos)
        self.assertEqual(set(state.proprio_record()), {"q", "gripper"})
        self.assertFalse(model.finger_geometry(state.q, current())["valid"])
        self.assertFalse(calibrated_fixture()[0].finger_geometry(state.q, None)["valid"])

    def test_named_proprio_is_copied_and_invalid_values_rejected(self):
        _, state = fixture(); positions = current()
        copied = replace(state, finger_qpos=positions)
        positions["left_j0"] = 0
        self.assertEqual(copied.finger_qpos["left_j0"], .05)
        record = copied.proprio_record(); record["finger_joint_positions_m"]["left_j0"] = 1
        self.assertEqual(copied.finger_qpos["left_j0"], .05)
        for value in ({}, [], {"": 0}, {"j": True}, {"j": [0]}, {"j": "0"}, {"j": np.nan}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                named_finger_positions(value)

    def test_independent_fingers_at_same_average_opening_are_not_conflated(self):
        model, state = calibrated_fixture()
        a = model.finger_geometry(state.q, current(right=(.01, .03)))
        b = model.finger_geometry(state.q, current(right=(.02, .02)))
        np.testing.assert_equal(a["arms"]["left"], b["arms"]["left"])
        for name in ("right_f0", "right_f1"):
            self.assertFalse(np.allclose(a["arms"]["right"][name]["T_base_link"],
                                         b["arms"]["right"][name]["T_base_link"]))
        self.assertTrue(a["not_contact_surface_or_holding_evidence"])

    def test_fk_with_rotated_eef_and_both_joint_directions(self):
        finger = FingerKinematics(finger_spec())
        T = transform([.3, -.2, .7], Rotation.from_euler("xyz", [.4, -.6, 1.]).as_quat())
        result = finger.geometry({a: T for a in ("left", "right")}, current((.015, .035), (.01, .02)))
        for arm, openings in (("left", (.015, .035)), ("right", (.01, .02))):
            for i, width in enumerate(openings):
                expected = T @ np.array([0, (1 if i == 0 else -1) * width, .02, 1])
                actual = np.asarray(result["arms"][arm][arm + "_f" + str(i)]["T_base_link"])
                np.testing.assert_allclose(actual[:3, 3], expected[:3], atol=1e-12)
                np.testing.assert_allclose(actual[:3, :3], T[:3, :3], atol=1e-12)

    def test_arm_fk_and_finger_fk_are_composed_once(self):
        model, state = calibrated_fixture(); q = state.q.copy(); q[11] = .02
        a = model.finger_geometry(state.q, current())
        b = model.finger_geometry(q, current())
        for arm in ("left", "right"):
            p0 = np.asarray(a["arms"][arm][arm + "_f0"]["T_base_link"])[:3, 3]
            p1 = np.asarray(b["arms"][arm][arm + "_f0"]["T_base_link"])[:3, 3]
            np.testing.assert_allclose(p1 - p0, [.02 if arm == "right" else 0, 0, 0], atol=1e-12)

    def test_reference_conversion_corrects_com_before_eef_transform(self):
        eef = transform([.4, -.1, .8], Rotation.from_euler("xyz", [.2, -.3, .7]).as_quat())
        link = transform([.5, .05, .9], Rotation.from_euler("xyz", [-.4, .2, -.1]).as_quat())
        rng = np.random.default_rng(73)
        screws = rng.normal(size=(6, 2)); com = np.array([.02, -.03, .04])
        # Independent native geometric J construction, including nonzero angular columns.
        w_base = eef[:3, :3] @ screws[3:]
        v_base = eef[:3, :3] @ screws[:3] + np.cross(eef[:3, 3], w_base.T).T
        at_com = v_base + np.cross(w_base.T, link[:3, 3] + link[:3, :3] @ com).T
        result = link_reference(eef, link, np.vstack([at_com, w_base]), com)
        np.testing.assert_allclose(result["screws"], screws, atol=1e-12)
        np.testing.assert_allclose(eef @ np.asarray(result["T_reference"]), link, atol=1e-12)

    def test_finite_difference_matches_named_prismatic_axis(self):
        f = FingerKinematics(finger_spec()); eef = {a: np.eye(4) for a in ("left", "right")}
        for arm in eef:
            for i, sign in enumerate((1, -1)):
                p = current((.02, .02), (.02, .02)); minus = dict(p); plus = dict(p)
                minus[arm + "_j" + str(i)] -= 1e-6; plus[arm + "_j" + str(i)] += 1e-6
                def point(q):
                    return np.asarray(f.geometry(eef, q)["arms"][arm][arm + "_f" + str(i)]["T_base_link"])[:3, 3]
                np.testing.assert_allclose((point(plus) - point(minus)) / 2e-6, [0, sign, 0], atol=1e-10)

    def test_missing_extra_and_outside_bounds_are_not_repaired(self):
        f = FingerKinematics(finger_spec()); eef = {a: np.eye(4) for a in ("left", "right")}
        for positions in (None, {"left_j0": .02}, {**current(), "other": 0},
                          current(right=(-.01, .02)), current(right=(.06, .02))):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                f.geometry(eef, positions)

    def test_bad_calibration_and_nonrigid_transforms_rejected(self):
        edits = [lambda s: s.update(version=True), lambda s: s.update(scene_truth=True), lambda s: s.update(embodiment="unknown"),
                 lambda s: s["arms"]["right"].update(joint_names=["left_j0", "left_j1"]),
                 lambda s: s["arms"]["right"]["links"]["right_f0"]["screws"][5].__setitem__(0, 1),
                 lambda s: s["arms"]["right"]["links"]["right_f0"]["screws"][1].__setitem__(0, 2),
                 lambda s: s["arms"]["right"]["links"]["right_f0"]["T_reference"][0].__setitem__(0, -1)]
        for edit in edits:
            spec = finger_spec(); edit(spec)
            with self.assertRaises(ValueError): FingerKinematics(spec)
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError):
            FingerKinematics(finger_spec()).geometry({"left": bad, "right": np.eye(4)}, current())

    def test_returned_geometry_and_original_spec_cannot_mutate_model(self):
        spec = finger_spec(); f = FingerKinematics(spec)
        eef = {a: np.eye(4) for a in ("left", "right")}
        expected = f.geometry(eef, current())
        spec["arms"]["left"]["links"]["left_f0"]["screws"][1][0] = 9
        changed = f.geometry(eef, current()); changed["arms"]["left"]["left_f0"]["T_base_link"][0][0] = 9
        self.assertEqual(f.geometry(eef, current()), expected)

    def test_actual_runner_state_preserves_fingers_only_when_opted_in(self):
        model, state = calibrated_fixture()
        path = Path(__file__).resolve().parents[2] / "scripts/semantic_robot/run_v2.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "state_now")
        code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")
        for enabled in (False, True):
            ns = {"kin": SimpleNamespace(state=lambda: state), "model": model,
                  "args": SimpleNamespace(finger_kinematics=enabled)}
            exec(code, ns); actual = ns["state_now"]()
            self.assertEqual(actual.finger_qpos, current() if enabled else None)
            np.testing.assert_array_equal(actual.q, state.q)
            np.testing.assert_array_equal(actual.gripper, state.gripper)

    def test_actual_reanchor_snapshot_preserves_independent_fingers_without_alias(self):
        from test_search_reanchor import SearchReanchorTests
        case = SearchReanchorTests(); case.setup()
        case.state.finger_qpos = current()
        result, snapshot = case.attempt()
        self.assertTrue(result["valid"])
        self.assertEqual(len(snapshot), 8); self.assertIsNone(snapshot[6])
        self.assertEqual(snapshot[7], case.state.finger_qpos)
        case.state.finger_qpos["left_j0"] = .02
        self.assertEqual(snapshot[7]["left_j0"], .05)


def native_fixture():
    model, state = calibrated_fixture()
    kin = CalibratedRobot.__new__(CalibratedRobot)
    kin.path = "robot"
    kin.links = {"left": "left_eef", "right": "right_eef"}
    poses = {arm + "_eef": state.poses[arm] for arm in ("left", "right")}
    names = list(current())
    joint = np.array([current()[name] for name in names])
    full_jac = np.zeros((1, 6, 6, 10))  # Two EEF links plus four fingers; floating base offset.
    link_indices = {"left_eef": 1, "right_eef": 2}
    start, end = {}, {}
    for arm_index, arm in enumerate(("left", "right")):
        for i, sign in enumerate((1, -1)):
            name = arm + "_f" + str(i)
            p = state.poses[arm][0] + [0, sign * .05, .02]
            poses[name] = (p, np.array([0, 0, 0, 1]))
            index = 3 + arm_index * 2 + i
            link_indices[name] = index
            full_jac[0, index - 1, 1, 6 + arm_index * 2 + i] = sign
            defs = [SimpleNamespace(link_name=name, position=np.array([0, 0, z])) for z in (-.01, .01)]
            (start if i == 0 else end)[arm] = defs
    kin.robot = SimpleNamespace(
        model="r1pro", has_end_effector_variants=False, joints=dict.fromkeys(names),
        get_joint_positions=lambda: joint.copy(), joint_lower_limits=np.zeros(4),
        joint_upper_limits=np.full(4, .05),
        finger_joint_names={a: [a + "_j0", a + "_j1"] for a in ("left", "right")},
        gripper_control_idx={"left": np.array([0, 1]), "right": np.array([2, 3])},
        finger_link_names={a: [a + "_f0", a + "_f1"] for a in ("left", "right")},
        assisted_grasp_start_points=start, assisted_grasp_end_points=end)
    kin.api = SimpleNamespace(get_all_relative_jacobians=lambda path: full_jac.copy(),
                              get_link_index=lambda path, link: link_indices[link],
                              get_link_relative_position_orientation=lambda path, link: poses[link])
    kin.local_com = lambda name: np.array([.01, -.02, .03])
    return kin, model, state, joint, full_jac


class NativeFingerExportTests(unittest.TestCase):
    def test_actual_export_has_named_positions_and_correct_robot_reference(self):
        kin, model, state, _, _ = native_fixture()
        spec = kin.finger_kinematics(state)
        geometry = FingerKinematics(spec).geometry(
            {a: transform(*state.poses[a]) for a in ("left", "right")}, current())
        expected = model.finger_geometry(state.q, current())
        for arm in ("left", "right"):
            for name in geometry["arms"][arm]:
                np.testing.assert_allclose(geometry["arms"][arm][name]["T_base_link"],
                                           expected["arms"][arm][name]["T_base_link"], atol=1e-12)
                np.testing.assert_allclose(geometry["arms"][arm][name]["asset_grasp_strip_base_m"],
                                           expected["arms"][arm][name]["asset_grasp_strip_base_m"], atol=1e-12)

    def test_native_export_rejects_stale_individual_joint_even_if_mean_unchanged(self):
        kin, _, state, joint, _ = native_fixture()
        joint[0] -= .001; joint[1] += .001
        with self.assertRaisesRegex(ValueError, "moved"):
            kin.finger_kinematics(state)

    def test_native_export_rejects_joint_name_or_cross_hand_coupling(self):
        kin, _, state, _, _ = native_fixture()
        kin.robot.finger_joint_names["left"].reverse()
        with self.assertRaisesRegex(ValueError, "names disagree"):
            kin.finger_kinematics(state)
        kin, _, state, _, jac = native_fixture()
        jac[0, 2, 1, 8] = .1
        with self.assertRaisesRegex(ValueError, "opposite hand"):
            kin.finger_kinematics(state)

    def test_native_export_rejects_finger_dependent_eef_and_unknown_hand(self):
        kin, _, state, _, jac = native_fixture()
        jac[0, 0, 1, 6] = 1
        with self.assertRaisesRegex(ValueError, "EEF"):
            kin.finger_kinematics(state)
        kin, _, state, _, _ = native_fixture(); kin.robot.model = "other"
        with self.assertRaisesRegex(ValueError, "supported R1Pro"):
            kin.finger_kinematics(state)

    def test_native_state_preserves_individual_positions_instead_of_repeating_mean(self):
        _, reference = fixture()
        kin = OGKinematics.__new__(OGKinematics)
        kin.path = "robot"; kin.indices = np.arange(18)
        kin.lower = reference.lower; kin.upper = reference.upper
        kin.links = {a: a for a in ("left", "right", "torso")}
        joints = np.r_[np.zeros(18), [.01, .03, .02, .04]]
        kin.robot = SimpleNamespace(get_joint_positions=lambda: joints.copy(),
            finger_joint_names={a: [a + "_j0", a + "_j1"] for a in ("left", "right")},
            gripper_control_idx={"left": np.array([18, 19]), "right": np.array([20, 21])},
            _get_proprioception_dict=lambda: {"gripper_left_qpos": np.array([.01, .03]),
                "gripper_right_qpos": np.array([.02, .04]), "base_qvel": np.zeros(3)})
        kin.api = SimpleNamespace(get_all_relative_jacobians=lambda path: np.zeros((1, 3, 6, 28)),
            get_link_relative_position_orientation=lambda path, name: reference.poses[name],
            get_link_index=lambda path, name: ("left", "right", "torso").index(name) + 1)
        state = kin.state()
        np.testing.assert_allclose(state.gripper, [.02, .03])
        self.assertEqual(state.finger_qpos, current((.01, .03), (.02, .04)))
        np.testing.assert_array_equal(state.q, joints[:18])


if __name__ == "__main__":
    unittest.main()

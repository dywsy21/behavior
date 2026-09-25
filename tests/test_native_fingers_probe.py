import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'src', ROOT/'scripts/semantic_robot', ROOT/'tests/semantic_robot'):
    sys.path.insert(0, str(path))
import probe_native_fingers as probe
from test_finger_kinematics import calibrated_fixture, finger_spec


def vectors():
    q = np.zeros(28)
    lower, upper = np.full(28, -1.), np.ones(28)
    lower[22:26], upper[22:26], q[22:26] = 0., .05, .05
    return q, lower, upper, np.arange(18), np.arange(22, 26)


def geometry_fixture():
    model, state = calibrated_fixture()
    geometry = model.finger_geometry(state.q, state.finger_qpos)
    poses = {}
    for arm, links in geometry['arms'].items():
        for name, row in links.items():
            T = np.asarray(row['T_base_link'])
            poses[name] = (T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat())
    kin = SimpleNamespace(path='robot', state=lambda: state,
        compare=lambda m: {name: {'position_m': 0., 'angle_rad': 0., 'jacobian_max_abs': 0.}
                           for name in ('left', 'right', 'torso')},
        api=SimpleNamespace(get_link_relative_position_orientation=lambda path, name: poses[name]))
    surfaces = {'links': {name: {'meshes': {'piece': {'vertices_link_m':
                       [[0., 0., 0.], [.04, 0., 0.], [0., .02, 0.], [0., 0., .03]]}}}
                         for name in poses}}
    return kin, model, surfaces, poses


class NativeFingerProbeTests(unittest.TestCase):
    def test_import_does_not_construct_app_or_import_gpu_framework(self):
        code = ('import sys; sys.path.insert(0, ' + repr(str(ROOT/'scripts/semantic_robot')) + '); '
                'import probe_native_fingers; '
                'assert not {"torch", "omnigibson", "isaacsim"}.intersection(sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True, timeout=10)

    def test_config_keeps_physics_controller_and_reset_without_sensors_or_task(self):
        original = {'model': 'r1pro', 'action_normalize': False, 'eval': {'unused': 1},
                    'obs_modalities': ['proprio', 'rgb'], 'self_collisions': True,
                    'grasping_mode': 'assisted', 'reset_joint_pos': [1., 2.],
                    'controller_config': {'base': {'kp': 150}},
                    'sensor_config': {'VisionSensor': {'sensor_kwargs': {'image_width': 1080}}}}
        before = copy.deepcopy(original)
        config = probe.robot_only_config(original)
        self.assertEqual(original, before)
        robot = config['robots'][0]
        for key in original.keys()-{'eval', 'obs_modalities'}:
            self.assertEqual(robot[key], original[key])
        self.assertEqual(robot['obs_modalities'], [])
        self.assertEqual(robot['include_sensor_names'], [])
        self.assertNotIn('eval', robot)
        self.assertEqual(config['task'], {'type': 'DummyTask'})
        self.assertEqual(config['objects'], [])
        self.assertEqual(config['env']['physics_frequency'], 120)
        self.assertEqual(config['env']['action_frequency'], 30)
        self.assertFalse(config['env']['automatic_reset'])

    def test_original_guard_and_frozen_single_use_profile_unchanged(self):
        base = probe.supervisor
        names = ('ENTRYPOINT', 'OUTPUT', 'RUNTIME', 'WALL_SECONDS', 'PROFILE_SETTINGS', 'PROFILE_APP_CONFIG',
                 'RUNTIME_SETTINGS', 'DEPENDENCIES', 'FIXED_ENV', 'SUCCESS_FIELDS', 'BUDGET_DETAILS')
        saved = {name: getattr(base, name) for name in names}
        try:
            probe.configure_profile()
            self.assertEqual(base.WALL_SECONDS, 600)
            self.assertEqual(base.AUXILIARY_MIB, 512)
            self.assertEqual(base.SUCCESS_FIELDS['completed_samples'], 9)
            self.assertEqual(base.BUDGET_DETAILS['model_calls'], 0)
            self.assertTrue(base.SUCCESS_FIELDS['not_success_rate'])
            self.assertEqual(base.ENTRYPOINT, Path(probe.__file__).resolve())
            self.assertEqual(base.DEPENDENCIES[probe.SURFACES], probe.SURFACE_SHA)
            base.validate_settings(base.RUNTIME_SETTINGS.copy())
            # Explicit original memory gates still reject reserve/cap violations.
            rows = {u: {'used_mib': 73664, 'free_mib': 7489,
                        'processes': [{'pid': p, 'type': 'C', 'used_mib': 73644}]}
                    for u, p in base.TRAINING.items()}
            base.check_resources(rows)
            bad = copy.deepcopy(rows); bad[base.MAIN_GPU]['free_mib'] = 3071
            with self.assertRaises(RuntimeError): base.check_resources(bad, rows, 123)
            bad = copy.deepcopy(rows); bad[base.MAIN_GPU]['used_mib'] += 4097
            with self.assertRaises(RuntimeError): base.check_resources(bad, rows, 123)
        finally:
            for name, value in saved.items(): setattr(base, name, value)

    def test_pose_schedule_is_fixed_uses_actual_indices_and_individual_limits(self):
        values = vectors(); q, lo, hi, body, fingers = values
        lo[22], hi[22] = -.002, .048
        rows = probe.pose_schedule(*values)
        self.assertEqual([n for n, _ in rows], [p[0] for p in probe.POSES])
        np.testing.assert_allclose(rows[3][1][fingers], [.008, .04, .025, .025])
        np.testing.assert_allclose(rows[6][1][body[[10, 17]]], [.12, -.12])
        np.testing.assert_array_equal(rows[0][1], rows[8][1])
        np.testing.assert_array_equal(q[22:26], [.05]*4)
        self.assertFalse(np.array_equal(rows[1][1][fingers], rows[5][1][fingers]))
        self.assertAlmostEqual(rows[1][1][fingers].mean(), rows[5][1][fingers].mean())

    def test_invalid_schedule_rejected_without_clipping(self):
        for index, change in ((0, lambda x: x.__setitem__(10, .95)),
                              (0, lambda x: x.__setitem__(0, np.nan)),
                              (3, lambda x: x.__setitem__(17, 10)),
                              (4, lambda x: x.__setitem__(0, 0)),
                              (4, lambda x: x.__setitem__(0, 28))):
            values = list(vectors()); change(values[index])
            with self.assertRaises(ValueError): probe.pose_schedule(*values)

    def test_coverage_uses_actual_named_fingers_not_only_mean(self):
        q, _, _, body, fingers = vectors()
        q[fingers] = [.01, .04, .04, .01]
        probe.validate_pose_coverage(q, q.copy(), body, fingers)
        slightly_moved = q.copy(); slightly_moved[fingers] += .001
        probe.validate_pose_coverage(q, slightly_moved, body, fingers)
        actual = q.copy(); actual[fingers] = .025
        with self.assertRaises(ValueError): probe.validate_pose_coverage(q, actual, body, fingers)
        actual = q.copy(); actual[10] += .01
        with self.assertRaises(ValueError): probe.validate_pose_coverage(q, actual, body, fingers)
        actual[10] = np.nan
        with self.assertRaises(ValueError): probe.validate_pose_coverage(q, actual, body, fingers)

    def test_native_comparison_records_transforms_vertices_and_no_contact_claim(self):
        kin, model, surfaces, _ = geometry_fixture()
        result = probe.compare_native(kin, model, surfaces)
        self.assertTrue(result['passed'])
        self.assertFalse(result['cooked_contact_verified'])
        self.assertEqual(len(result['finger_links']), 4)
        for row in result['finger_links'].values():
            self.assertEqual(row['vertices'], 4)
            self.assertLess(row['authored_vertex_max_error_m'], 1e-10)

    def test_wrong_native_position_rotation_or_asset_link_fails(self):
        for mode in ('position', 'rotation', 'body'):
            kin, model, surfaces, poses = geometry_fixture()
            if mode == 'position': poses['right_f0'][0][0] += .001
            elif mode == 'rotation':
                p, q = poses['right_f0']
                poses['right_f0'] = (p, (Rotation.from_rotvec([0, 0, .01])*Rotation.from_quat(q)).as_quat())
            else:
                original = kin.compare
                def changed(m):
                    row = original(m); row['torso']['position_m'] = .001; return row
                kin.compare = changed
            self.assertFalse(probe.compare_native(kin, model, surfaces)['passed'])
        del surfaces['links']['right_f0']
        with self.assertRaises(ValueError): probe.compare_native(kin, model, surfaces)

    def test_actual_sample_loop_calibrates_once_measures_nine_and_stops_on_failure(self):
        import torch
        for failing in (False, True):
            q, lo, hi, body, fingers = vectors()
            robot = SimpleNamespace(q=torch.as_tensor(q), joints=list(range(28)),
                joint_lower_limits=lo, joint_upper_limits=hi,
                gripper_control_idx={'left': fingers[:2], 'right': fingers[2:]})
            robot.get_joint_positions = lambda: robot.q.clone()
            robot.set_joint_positions = lambda value, **kw: setattr(robot, 'q', value.clone())
            robot.set_joint_velocities = lambda value: None
            kin, initial, surfaces, _ = geometry_fixture()
            initial_state = kin.state()
            kin.indices = body
            kin.receipt = lambda: {'native_action_dim': 23}
            kin.finger_kinematics = lambda state: finger_spec()
            def state_now():
                positions = dict(zip(initial_state.finger_qpos, robot.q[fingers].numpy().tolist()))
                return initial.state(robot.q[body].numpy(), np.array([.025, .025]), np.zeros(3), positions)
            kin.state = state_now
            steps = []
            def step():
                steps.append(1)
                if failing and len(steps) == 4: robot.q[fingers] = .025
            sim = SimpleNamespace(step_physics=step)
            record = dict(calibration_count=0, completed_samples=0, joint_pose_sets=0, explicit_physics_steps=0)
            files = {}
            def save(name, value): files[name] = copy.deepcopy(value)
            # The loop is real; geometry itself has separate independent error tests.
            with patch('semantic_robot.v2.og_calibration.CalibratedRobot', return_value=kin), \
                    patch.object(probe, 'compare_native', return_value={'passed': True}) as compare:
                if failing:
                    with self.assertRaises(ValueError): probe.run_samples(robot, sim, surfaces, record, save)
                    self.assertEqual(record['completed_samples'], 3)
                    self.assertEqual(len(files['samples.json']), 4)
                    self.assertNotIn('native_geometry_passed', record)
                else:
                    probe.run_samples(robot, sim, surfaces, record, save)
                    self.assertTrue(record['native_geometry_passed'])
                    self.assertEqual(record['completed_samples'], 9)
                    self.assertEqual(len(steps), 9)
                    self.assertEqual(compare.call_count, 9)
                self.assertEqual(record['calibration_count'], 1)


if __name__ == '__main__': unittest.main()

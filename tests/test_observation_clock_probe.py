import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
import probe_observation_clock as probe
import launch_h72 as launch


class ObservationClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.position = np.array([10., 20., 30.])
        self.robot = SimpleNamespace(get_position_orientation=lambda: (self.position, [0., 0., 0., 1.]))
        self.sim = SimpleNamespace(current_time=1., current_time_step_index=120,
            get_physics_dt=lambda: 1/120, get_rendering_dt=lambda: 1/30, get_sim_step_dt=lambda: 1/30)
        self.sample = {'rgb': np.zeros((3, 4, 4), np.uint8), 'depth_linear': np.ones((3, 4), np.float32)}
        self.sensor = SimpleNamespace(get_obs=lambda: (self.sample, {}))

    def journal(self):
        value = probe.Journal(self.root); value.robot = self.robot
        value.read_poses = self.pose_reader(self.robot)
        self.addCleanup(value.close); return value

    @staticmethod
    def pose_reader(robot):
        return lambda: {'physx': robot.get_position_orientation(), 'fabric': robot.get_position_orientation()}

    def rows(self):
        return [json.loads(x) for x in (self.root/'PRIVILEGED_clock_pose.jsonl').read_text().splitlines()]

    def test_relative_robot_pose_not_absolute_and_actual_clock_recorded(self):
        j = self.journal(); j.record('a', self.sim)
        self.position[0] -= .06; self.sim.current_time += .2; self.sim.current_time_step_index += 24
        j.record('b', self.sim)
        rows = self.rows()
        np.testing.assert_allclose(rows[0]['diagnostic_physx_body_in_initial_body'], np.eye(4))
        self.assertAlmostEqual(rows[1]['diagnostic_physx_body_in_initial_body'][0][3], -.06)
        self.assertEqual(rows[1]['physics_index'], 144)
        self.assertAlmostEqual(rows[1]['simulation_time']-rows[0]['simulation_time'], .2)
        self.assertTrue(rows[1]['never_sent_to_actor_or_odometer'])

    def test_quaternion_rotation_is_in_original_body_frame(self):
        self.robot.get_position_orientation = lambda: (self.position, [0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
        j = self.journal(); j.record('a', self.sim); self.position[1] += .1; j.record('b', self.sim)
        np.testing.assert_allclose(np.array(self.rows()[1]['diagnostic_physx_body_in_initial_body'])[:3, 3], [.1,0,0], atol=1e-12)

    def test_duplicate_buffers_are_recorded_not_claimed_fresh(self):
        j = self.journal(); j.record('a', self.sim, head=self.sensor)
        self.sim.current_time += .2; self.position[0] += .03
        j.record('b', self.sim, head=self.sensor)
        a,b = self.rows(); self.assertEqual(a['head'], b['head'])
        self.assertNotEqual(a['diagnostic_physx_body_in_initial_body'], b['diagnostic_physx_body_in_initial_body'])
        self.assertNotIn('same_sensor_current_render', b)

    def test_head_buffers_are_hashed_before_potential_gpu_pose_sync(self):
        j = self.journal(); calls = []
        def sensor_read():
            calls.append('head'); return self.sample, {}
        def poses_read():
            calls.append('pose'); return self.pose_reader(self.robot)()
        self.sensor.get_obs=sensor_read; j.read_poses=poses_read
        j.record('a', self.sim, head=self.sensor)
        self.assertEqual(calls,['head','pose'])
        self.assertGreaterEqual(self.rows()[0]['head_read_hash_seconds'],0)
        self.assertGreaterEqual(self.rows()[0]['pose_read_seconds'],0)
        self.assertTrue(j.finish()['instrumented_timing_not_identical_to_H71'])

    def test_head_hash_drops_alpha_without_mutation(self):
        before = copy.deepcopy(self.sample); a = probe.head_hash(self.sensor)
        self.sample['rgb'][..., 3] = 123
        self.assertEqual(a, probe.head_hash(self.sensor))
        np.testing.assert_array_equal(self.sample['depth_linear'], before['depth_linear'])
        self.sample['depth_linear'][0,0] = 2
        self.assertNotEqual(a['depth_sha256'], probe.head_hash(self.sensor)['depth_sha256'])

    def test_digest_separate_from_normal_and_binds_both_diagnostic_files(self):
        self.assertNotEqual(probe.audit_digest('a'), 'a')
        self.assertNotEqual(probe.audit_digest('a'), probe.audit_digest('b'))
        original = Path.read_bytes
        for name in ('probe_observation_clock.py', 'launch_h72.py'):
            with patch.object(Path, 'read_bytes', lambda p: original(p)+(b' ' if p.name == name else b'')):
                changed = probe.audit_digest('a')
            self.assertNotEqual(changed, probe.audit_digest('a'))

    def test_finite_pose_and_bounded_journal(self):
        j = self.journal(); self.position[0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Finite'): j.record('bad', self.sim)
        self.position[0] = 10; j.rows = 8192
        with self.assertRaisesRegex(ValueError, 'Bounded'): j.record('bad', self.sim)

    def test_installed_clock_source_checked_in_parent_and_worker(self):
        path = self.root/'context.py'; path.write_bytes(b'known')
        with patch.dict(probe.INSTALLED, {path:hashlib.sha256(b'known').hexdigest()}, clear=True):
            probe.validate_installed(); path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'implementation changed'): probe.validate_installed()
        with patch.object(launch, 'validate_installed', side_effect=ValueError('changed')):
            with self.assertRaisesRegex(ValueError, 'changed'): launch.identity(None)

    def test_physx_reader_uses_actual_named_base_not_fabric_or_fixed_root(self):
        raw = np.array([[[0,0,0,0,0,0,1], [1,2,3,0,0,0,1]]], dtype=float)
        view = SimpleNamespace(_idx={'robot':0}, _link_idx=[{'base':1}],
                               _view=SimpleNamespace(get_link_transforms=lambda: raw))
        module = ModuleType('omnigibson.utils.usd_utils')
        module.ControllableObjectViewAPI = SimpleNamespace(_VIEWS_BY_PATTERN={'pattern':view})
        module.get_robot_kinematic_tree_pattern = lambda path: 'pattern'
        self.robot.articulation_root_path='robot'; self.robot.base_footprint_link_name='base'
        with patch.dict(sys.modules, {'omnigibson.utils.usd_utils':module}):
            read = probe.native_pose_reader(self.robot)
            a = read(); raw[0,1,0] += .05; b = read()
        np.testing.assert_array_equal(a['physx'][0], [1,2,3])
        self.assertAlmostEqual(b['physx'][0][0]-a['physx'][0][0], .05)
        np.testing.assert_array_equal(a['fabric'][0], b['fabric'][0])
        self.assertFalse(hasattr(view, '_read_cache'))

    def runner(self, renders=4):
        parent = self
        state = object(); result = ({'public': True}, {}, {'original_receipt': True})
        class Robot:
            def __init__(self): self.robot = parent.robot
            def state(self): return state
        class Camera:
            def __init__(self): self.sensors = {'head': parent.sensor}; self.snapshot_id = 0
            def read(self, model, *, render):
                for _ in range(renders): render()
                self.snapshot_id += 1
                return result
        def write(path, value): Path(path).write_text(json.dumps(value))
        return SimpleNamespace(CalibratedRobot=Robot, OnboardRGBD=Camera,
                               write=write, implementation_digest=lambda: 'original'), state, result

    def test_instrumentation_returns_original_objects_and_exact_four_renders(self):
        runner, state, images = self.runner()
        j = probe.instrument(runner, self.root, lambda: self.sim, self.pose_reader); self.addCleanup(j.close)
        robot = runner.CalibratedRobot(); camera = runner.OnboardRGBD()
        self.assertIs(robot.state(), state); self.assertIs(robot.state(), state)
        self.assertEqual(j.rows, 1)  # Duplicate proprio reads don't flood journal.
        renders = []
        self.assertIs(camera.read(None, render=lambda: renders.append(1)), images)
        self.assertEqual(len(renders), 4)
        self.assertEqual([r['render_index'] for r in self.rows() if r['phase']=='capture_after_render'], [1,2,3,4])
        self.assertEqual(runner.implementation_digest(), probe.audit_digest('original'))
        raw = {'gate_ok': False, 'controls': 18}
        runner.write(self.root/'result.json', raw)
        self.assertEqual(json.loads((self.root/'result.json').read_text()), raw)
        audit = json.loads((self.root/'clock_audit.json').read_text())
        self.assertFalse(audit['truth_sent_to_actor_or_odometer'])
        self.assertEqual(audit['journal_sha256'], hashlib.sha256((self.root/'PRIVILEGED_clock_pose.jsonl').read_bytes()).hexdigest())

    def test_changed_barrier_and_original_render_errors_are_not_swallowed(self):
        for n in (3,):
            runner, _, _ = self.runner(n)
            j = probe.instrument(runner, self.root, lambda: self.sim, self.pose_reader); self.addCleanup(j.close)
            runner.CalibratedRobot()
            with self.assertRaisesRegex(ValueError, 'four-render'): runner.OnboardRGBD().read(None, render=lambda: None)

    def test_failure_is_sealed_inside_write_before_native_exit(self):
        runner, _, _ = self.runner()
        j = probe.instrument(runner, self.root, lambda: self.sim, self.pose_reader); self.addCleanup(j.close)
        runner.CalibratedRobot().state()
        error = {'error':'original render failed'}
        runner.write(self.root/'failure.json', error)
        audit = json.loads((self.root/'clock_audit_failure.json').read_text())
        self.assertFalse(audit['complete']); self.assertEqual(audit['events'],1)
        self.assertEqual(audit['original_failure'], error)
        self.assertFalse((self.root/'clock_audit.json').exists())
        with patch.object(j, 'finish', side_effect=OSError('secondary seal failed')):
            runner.write(self.root/'failure.json', error)
        self.assertEqual(json.loads((self.root/'failure.json').read_text()),error)

    def test_launcher_does_not_change_gate_controls_flags_or_resources(self):
        base = SimpleNamespace(REPO=self.root, PYTHON=Path('/python'))
        original = launch.COMMAND(base); audited = launch.command(base)
        self.assertEqual(original[:1]+original[2:], audited[:1]+audited[2:])
        self.assertTrue(audited[1].endswith('probe_observation_clock.py'))

    def test_diagnostic_completion_does_not_turn_failed_gate_into_pass(self):
        folder = self.root/'gate'; folder.mkdir()
        j = probe.Journal(folder); j.robot=self.robot; self.addCleanup(j.close)
        j.read_poses = self.pose_reader(self.robot)
        source = folder/'decision_011'; source.mkdir()
        for snapshot, control in enumerate((213,219,225,231), 1):
            j.record('capture_before', self.sim, head=self.sensor, snapshot=snapshot)
            for i in range(1,5): j.record('capture_after_render', self.sim, head=self.sensor, snapshot=snapshot, render_index=i)
            j.record('capture_return', self.sim, head=self.sensor, snapshot=snapshot)
            target = source if control==213 else source/'motion_substeps'/f'control_{control:06d}'
            target.mkdir(parents=True,exist_ok=True)
            (target/'depth_receipt.json').write_text(json.dumps({'head':{'snapshot_id':snapshot,**probe.head_hash(self.sensor)}}))
        (source/'action_motion.json').write_text(json.dumps({'control_start':213,'control_end':231,
            'segments':[{'control_start':n,'control_end':n+6} for n in (213,219,225)]}))
        j.record('final_after_safe_hold', self.sim)
        (folder/'clock_audit.json').write_text(json.dumps(j.finish()))
        result = {'status':'complete', 'task':0, 'prefix_controls':0, 'diagnostic_replay_controls':0,
                  'native_profile':'a100_full_v1', 'implementation_digest':probe.audit_digest('normal'),
                  'model_calls':0, 'full_task_success_rate_claim':False, 'controls':232,
                  'decisions':[{} for _ in range(12)], 'gate_ok':False}
        result['decisions'][11] = {'decision':11,'action':{'part':'base','move':'back','scale':'fine','frame':'base'},
            'accepted_before_motion':True,'control_start':213,'control_end':231,'feedback':{'control_ticks':18}}
        original=copy.deepcopy(result)
        with patch.object(launch.gate, 'ROOT', self.root):
            launch.validate_result(result, 'normal'); self.assertEqual(result, original)
            for field,value in [('model_calls',1),('prefix_controls',1),('decisions',[]),('controls',1537),
                                ('implementation_digest','normal'),('full_task_success_rate_claim',True)]:
                with self.subTest(field=field),self.assertRaises(ValueError):
                    launch.validate_result(result | {field:value}, 'normal')
            rejected = copy.deepcopy(result); rejected['decisions'][11]['accepted_before_motion']=False
            with self.assertRaisesRegex(ValueError,'not actually executed'): launch.validate_result(rejected,'normal')
            path=source/'depth_receipt.json'; receipt=json.loads(path.read_text())
            receipt['head']['snapshot_id']=999; path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'capture sequence'): launch.validate_result(result,'normal')
            receipt['head']['snapshot_id']=1; path.write_text(json.dumps(receipt))
            with (folder/'PRIVILEGED_clock_pose.jsonl').open('ab') as f: f.write(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete'): launch.validate_result(result, 'normal')


if __name__ == '__main__': unittest.main()

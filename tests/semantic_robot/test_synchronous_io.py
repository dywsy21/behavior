from contextlib import contextmanager
import ast
import copy
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from semantic_robot.v2.synchronous_io import SynchronousIO, reference_seconds, attach_references, PLAY_SIMULATIONS, VIEWS


class FakeSim:
    current_time=1.
    current_time_step_index=120
    rendering=True
    def get_physics_dt(self):return 1/120
    def get_sim_step_dt(self):return 1/30
    def get_rendering_dt(self):return 1/30
    @contextmanager
    def render_on_step(self, value):
        old=self.rendering;self.rendering=value
        try:yield
        finally:self.rendering=old
    def render(self):pass
    def advance(self, ticks=4):
        self.current_time+=ticks/120;self.current_time_step_index+=ticks


class SynchronousIOTests(unittest.TestCase):
    def setUp(self):
        self.sim=FakeSim();self.rows=[];self.settings={PLAY_SIMULATIONS:True}
        self.api=SimpleNamespace(get=self.settings.get,set=lambda k,v:self.settings.__setitem__(k,v))
        self.refs={v:Mock(get_data=lambda:{'referenceTimeNumerator':120,'referenceTimeDenominator':120}) for v in VIEWS}
        self.capture=Mock()
        self.io=SynchronousIO(self.sim,self.refs,self.capture,self.api,lambda row:self.rows.append(copy.deepcopy(row)))

    def test_physics_only_exact_step_even_when_render_requested(self):
        with self.io.control(render_requested=True):
            self.assertFalse(self.sim.rendering);self.sim.advance()
        self.assertTrue(self.sim.rendering);self.assertEqual(self.io.controls,1)
        self.assertTrue(self.rows[0]['completed']);self.assertFalse(self.rows[0]['actual_render_on_step'])
        self.assertTrue(self.rows[0]['render_requested']);self.capture.assert_not_called()

    def test_zero_double_and_time_only_bad_steps_fail_closed(self):
        for ticks in (0,1,8):
            with self.subTest(ticks=ticks),self.assertRaisesRegex(RuntimeError,'four physics'):
                with self.io.control(render_requested=False):self.sim.advance(ticks)
        self.assertEqual(self.io.controls,0)
        self.assertEqual([r['call'] for r in self.rows],[1,2,3])
        self.assertTrue(all(not r['completed'] for r in self.rows))
        with self.assertRaises(RuntimeError):
            with self.io.control(render_requested=False):self.sim.current_time_step_index+=4

    def test_rates_rechecked_before_issuing_a_control(self):
        self.sim.get_sim_step_dt=lambda:1/60
        with self.assertRaises(ValueError):
            with self.io.control(render_requested=False):self.fail('Must not issue a command')

    def test_original_env_error_retained_and_render_context_restored(self):
        with self.assertRaisesRegex(RuntimeError,'original'):
            with self.io.control(render_requested=True):
                self.sim.advance();raise RuntimeError('original')
        self.assertTrue(self.sim.rendering);self.assertFalse(self.rows[-1]['completed'])
        self.assertEqual(self.rows[-1]['after']['physics_index'],124)

    def test_matching_three_camera_times_and_zero_physics_capture(self):
        def capture(**kwargs):self.assertFalse(self.settings[PLAY_SIMULATIONS])
        self.capture.side_effect=capture
        receipt=self.io.synchronize()
        self.capture.assert_called_once_with(delta_time=0.,pause_timeline=False,wait_for_render=True,rt_subframes=4)
        self.assertEqual(set(receipt),VIEWS);self.assertEqual(self.sim.current_time_step_index,120)
        self.assertTrue(self.settings[PLAY_SIMULATIONS]);self.assertTrue(self.rows[-1]['completed'])
        # Repeated static-state captures are valid; no image-difference rule.
        self.io.synchronize();self.assertEqual(self.io.captures,2)

    def test_one_stale_camera_rejects_all_and_preserves_evidence(self):
        self.refs['right_wrist'].get_data=lambda:{'referenceTimeNumerator':119,'referenceTimeDenominator':120}
        with self.assertRaisesRegex(RuntimeError,'ReferenceTime'):self.io.synchronize()
        self.assertEqual(self.io.captures,0);self.assertTrue(self.settings[PLAY_SIMULATIONS])
        self.assertAlmostEqual(self.rows[-1]['reference_seconds']['right_wrist'],119/120)

    def test_capture_that_steps_physics_is_rejected(self):
        self.capture.side_effect=lambda **kwargs:self.sim.advance()
        with self.assertRaisesRegex(RuntimeError,'advanced physics'):self.io.synchronize()
        self.assertTrue(self.settings[PLAY_SIMULATIONS]);self.assertFalse(self.rows[-1]['completed'])

    def test_original_render_only_call_cannot_advance_physics(self):
        self.sim.render=self.sim.advance
        with self.assertRaisesRegex(RuntimeError,'Render-only'):self.io.synchronize()
        self.capture.assert_not_called();self.assertTrue(self.settings[PLAY_SIMULATIONS])

    def test_reference_fraction_strict_and_native_integer_supported(self):
        self.assertEqual(reference_seconds({'referenceTimeNumerator':np.uint64(30),'referenceTimeDenominator':np.uint64(30)}),1.)
        for value in (None,{}, {'referenceTimeNumerator':1,'referenceTimeDenominator':0},
                      {'referenceTimeNumerator':True,'referenceTimeDenominator':1},
                      {'referenceTimeNumerator':-1,'referenceTimeDenominator':1},
                      {'referenceTimeNumerator':1.,'referenceTimeDenominator':1}):
            with self.subTest(value=value),self.assertRaises(ValueError):reference_seconds(value)

    def test_capture_error_restores_flag_and_preserves_primary_failure(self):
        self.capture.side_effect=RuntimeError('native failed')
        self.io.record=Mock(side_effect=OSError('disk also failed'))
        with self.assertRaisesRegex(RuntimeError,'native failed'):self.io.synchronize()
        self.assertTrue(self.settings[PLAY_SIMULATIONS])

    def test_invalid_failure_clock_never_replaces_primary(self):
        with self.assertRaisesRegex(RuntimeError,'PRIMARY_CONTROL'):
            with self.io.control(render_requested=True):
                self.sim.current_time=float('nan');raise RuntimeError('PRIMARY_CONTROL')
        self.assertIn('after_clock_error',self.rows[-1])
        self.sim.current_time=1.
        def fail(**kwargs):
            self.sim.current_time=float('nan');raise RuntimeError('PRIMARY_CAPTURE')
        self.capture.side_effect=fail
        with self.assertRaisesRegex(RuntimeError,'PRIMARY_CAPTURE'):self.io.synchronize()
        self.assertIn('after_clock_error',self.rows[-1])

    def test_restore_error_preserves_capture_error_but_blocks_success(self):
        setter=self.api.set
        def set_fail(key,value):
            if value is True:raise OSError('restore failed')
            setter(key,value)
        self.api.set=set_fail
        self.capture.side_effect=RuntimeError('PRIMARY_CAPTURE')
        with self.assertRaisesRegex(RuntimeError,'PRIMARY_CAPTURE'):self.io.synchronize()
        self.assertIn('restore failed',self.rows[-1]['restore_error'])
        self.settings[PLAY_SIMULATIONS]=True;self.capture.side_effect=None
        with self.assertRaisesRegex(OSError,'restore failed'):self.io.synchronize()
        self.assertFalse(self.rows[-1]['completed'])

    def test_unknown_flag_or_missing_camera_cannot_be_used(self):
        self.settings[PLAY_SIMULATIONS]=None
        with self.assertRaises(ValueError):self.io.synchronize()
        self.capture.assert_not_called()
        with self.assertRaises(ValueError):SynchronousIO(self.sim,{},self.capture,self.api,self.rows.append)

    def test_real_runner_step_uses_one_physics_only_issue_and_preserves_grips(self):
        source=Path(__file__).resolve().parents[2]/'scripts/semantic_robot/run_v2.py'
        node=copy.deepcopy(next(n for n in ast.walk(ast.parse(source.read_text()))
                                if isinstance(n,ast.FunctionDef) and n.name=='step'))
        node.body=[ast.Global(names=n.names) if isinstance(n,ast.Nonlocal) else n for n in node.body]
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec')
        for ticks,extra in ((4,0),(0,0),(4,4)):
            calls=[];manager=Mock();action=np.zeros(23);action[[14,22]]=[-1,1]
            def issue(value,**kwargs):
                self.assertFalse(self.sim.rendering);manager.issued.assert_called_once_with(value)
                calls.append(value);self.sim.advance(ticks)
                return {},0,False,False,{}
            def postprocess(value):
                self.assertFalse(self.sim.rendering)
                if extra:self.sim.advance(extra)
                return value
            scope={'native_io':self.io,'np':np,'grounded':True,'manager':manager,'last_issued_grips':None,
                   'info':{},'terminal':False,'env':SimpleNamespace(step=issue),
                   'environment':SimpleNamespace(evaluator=SimpleNamespace(_preprocess_obs=lambda x:x,
                                                                           _sync_lights_and_get_obs=postprocess))}
            exec(code,scope)
            if ticks and not extra:self.assertFalse(scope['step'](action,render=True))
            else:
                with self.assertRaisesRegex(RuntimeError,'four physics'):scope['step'](action)
            self.assertEqual(len(calls),1)
            np.testing.assert_array_equal(scope['last_issued_grips'],[-1,1])

    def test_close_once_and_error_preservation(self):
        self.io.release=Mock();self.io.close();self.io.close()
        self.io.release.assert_called_once();self.assertEqual(self.rows[-1]['kind'],'close')
        self.assertTrue(self.rows[-1]['completed'])
        with self.assertRaisesRegex(RuntimeError,'closed'):
            with self.io.control(render_requested=True):self.fail('closed')
        io=SynchronousIO(self.sim,self.refs,self.capture,self.api,self.rows.append,
                         Mock(side_effect=OSError('detach failed')))
        with self.assertRaisesRegex(RuntimeError,'PRIMARY'):
            try:raise RuntimeError('PRIMARY')
            finally:io.close()
        self.assertFalse(self.rows[-1]['completed']);self.assertIn('detach failed',self.rows[-1]['error'])
        io.close();io.release.assert_called_once()

    def test_partial_attach_and_detach_failures_keep_original_and_cleanup_all(self):
        sensors={v:SimpleNamespace(render_product='rp_'+v) for v in ('head','left_wrist','right_wrist')}
        annos=[Mock(),Mock(),Mock()];annos[1].attach.side_effect=RuntimeError('attach failed')
        annos[0].detach.side_effect=OSError('detach also failed')
        with self.assertRaisesRegex(RuntimeError,'attach failed'):
            attach_references(sensors,Mock(side_effect=annos))
        annos[0].detach.assert_called_once_with(['rp_head'])
        annos[1].detach.assert_called_once_with(['rp_left_wrist'])
        annos[2].attach.assert_not_called()
        annos=[Mock(),Mock(),Mock()]
        refs,release=attach_references(sensors,Mock(side_effect=annos))
        self.assertEqual(set(refs),VIEWS);release()
        for view,anno in zip(sensors,annos):anno.detach.assert_called_once_with([sensors[view].render_product])

    def test_onboard_waits_before_reading_and_binds_native_receipt(self):
        from semantic_robot.v2.onboard import OnboardRGBD
        from test_v2 import fixture
        model,_=fixture();adapter=OnboardRGBD.__new__(OnboardRGBD)
        ready=[]
        class Sensor:
            def get_obs(self):
                if not ready:raise AssertionError('Pixels read before synchronous barrier')
                return {'rgb':np.zeros((100,100,3),np.uint8),'depth_linear':np.ones((100,100),np.float32)},{}
        adapter.sensors={v:Sensor() for v in VIEWS};adapter.snapshot_id=0
        def synchronize():
            result=self.io.synchronize();ready.append(True);return result
        render=Mock()
        images,depths,receipt=adapter.read(model,render=render,synchronize=synchronize)
        render.assert_not_called()
        for v in VIEWS:
            self.assertEqual(receipt[v]['freshness_proof'],'native_reference_time')
            self.assertEqual(receipt[v]['native_time']['physics_index'],120)
            self.assertEqual(receipt[v]['render_barrier_updates'],1)


if __name__=='__main__':unittest.main()

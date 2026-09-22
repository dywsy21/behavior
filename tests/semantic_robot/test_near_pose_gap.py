import argparse
import ast
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.near_pose_gap import eligible, qualification, execution_check, execution_sweep
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action, ROTATIONS, TRANSLATIONS
from semantic_robot.v2.servo import SafeServo, ServoLimits
from semantic_robot.v2.wall_budget import WallTimeBudgetReached
from test_hand_body_collision import calibrated_fixture
from test_v2 import evidence


def setup(enabled=True):
    model,state,geometry=calibrated_fixture()
    model.spec['metadata']['grasp_region_reference_gripper_m']=[.05,.05]
    h=GroundedHarness([Goal('pick','object','right','held')],approach_reorientation=True,near_pose_gap=enabled)
    h.stage='APPROACH';h.last_gripper=state.gripper.copy();h.observation=evidence(hazard='occluded')
    h.grounding={'valid':True,'point_base_m':(model.grasp_centers(state.q)['right']+[.09,0,0]).tolist(),
                 'distance_to_active_closing_center_m':.09,'near_contact_review':{'confirmed_current_contact':True}}
    servo=SafeServo(model,state,[1,1],ServoLimits(robot_geometry_guards=True))
    c=GroundedController(model,servo,h);c.target=h.grounding;c.centers=model.grasp_centers(state.q)
    c.depth_guard=SimpleNamespace(check=lambda *a:(True,'FIXTURE_ONLY'),receipt=lambda:{},depth_images={})
    c.near_pose_geometry=geometry
    return model,state,h,servo,c


ORIGINAL_BEGIN=SafeServo.begin
def block_improving(trial,action,state,carry=False):
    if action.part=='right' and action.move=='forward' and action.frame in ('base','tool'):
        return trial.abort('UNREACHABLE_OR_COLLISION_BLOCKED')
    return ORIGINAL_BEGIN(trial,action,state,carry)


class FakeSweep:
    safe=True
    calls=[]
    def __init__(self,*args): self.args=args
    def check(self,plan):
        self.calls.append((self.args,plan.copy()))
        return self.safe,{'reason':'FIXTURE_ONLY_PASS' if self.safe else 'OBSERVED_FREE_ARM_SWEEP_OBSTACLE',
                          'unobserved_space_not_certified':True}


class NearPoseTests(unittest.TestCase):
    def candidates(self,c,state,safe=True):
        FakeSweep.safe=safe;FakeSweep.calls=[]
        with patch.object(SafeServo,'begin',block_improving), \
             patch('semantic_robot.v2.grounded_harness.observed_cloud',return_value=np.ones((50,3))), \
             patch('semantic_robot.v2.arm_observation_guard.ObservingArmGuard',FakeSweep):
            return c.candidates(state)

    def test_default_and_only_original_gap(self):
        _,_,h,_,_=setup(False);self.assertFalse(eligible(h))
        self.assertFalse(any(a.part == 'right' and a.move in ROTATIONS for a in h.palette()))
        for distance,wanted in ((.08,False),(.080001,True),(.09,True),(.10,True),(.100001,False),(np.nan,False)):
            _,_,h,_,_=setup();h.grounding['distance_to_active_closing_center_m']=distance
            self.assertEqual(eligible(h),wanted)
        with self.assertRaises(ValueError):
            GroundedHarness([Goal('pick','object','right','held')],near_pose_gap=True)

    def test_unknown_load_history_danger_and_unconfirmed_contact_fail_closed(self):
        for name in ('pending_grasp','hold_verified','possible_contact_after_close','workspace_close_seen'):
            for value in (None,{}, {'left':False}, {'left':False,'right':None}, {'left':False,'right':0}, {'left':False,'right':True}):
                _,_,h,_,_=setup();setattr(h,name,value)
                self.assertFalse(eligible(h))
        for mutate in (lambda h:setattr(h,'observation',evidence(hazard='collision')),
                       lambda h:setattr(h,'observation',evidence(enclosed=True)),
                       lambda h:setattr(h,'observation',None),
                       lambda h:setattr(h,'stage','ALIGN'),
                       lambda h:setattr(h,'stop_reason','STOP'),
                       lambda h:h.grounding.update(valid=False),
                       lambda h:h.grounding['near_contact_review'].update(confirmed_current_contact=False),
                       lambda h:h.last_gripper.__setitem__(0,.02)):
            _,_,h,_,_=setup();mutate(h);self.assertFalse(eligible(h))

    def test_calibration_actual_open_and_latches_mandatory(self):
        for mutate in (lambda m,s,g:m.spec['metadata'].pop('grasp_region_reference_gripper_m'),
                       lambda m,s,g:m.spec['metadata']['grasp_region_reference_fully_open'].update(right=False),
                       lambda m,s,g:s.gripper.__setitem__(1,.03),
                       lambda m,s,g:g.__setitem__(1,-1),
                       lambda m,s,g:g.__setitem__(1,1.1),
                       lambda m,s,g:g.__setitem__(0,np.nan)):
            m,s,h,v,_=setup();mutate(m,s,v.grips)
            self.assertFalse(qualification(h,s,v.grips,m,v.limits)['eligible'])

    def test_blocked_near_state_gets_only_bounded_checked_options(self):
        m,s,h,v,c=setup();q=s.q.copy();allowed=self.candidates(c,s)
        receipt=h.candidate_receipt;extra=[r for r in receipt['tested'] if r.get('near_pose_gap_option')]
        self.assertTrue(extra);self.assertLessEqual(len(extra),15)
        self.assertEqual(receipt['near_pose_gap']['additional_preflights'],len(extra))
        self.assertTrue(any(r['action']['move'] in ROTATIONS and r['accepted'] for r in extra))
        self.assertTrue(any(r['action']['move'] in TRANSLATIONS and r['accepted'] for r in extra))
        for r in extra:
            self.assertEqual(Action(**r['action']) in allowed,r['accepted'])
            if r['accepted']:self.assertTrue(r['near_pose_sweep']['safe'])
        self.assertTrue(all(a.part=='right' and a.scale in ('fine','micro') for a in allowed if a.move in ROTATIONS))
        self.assertFalse(any(a.move in ROTATIONS and a.scale=='coarse' for a in allowed))
        np.testing.assert_array_equal(s.q,q);self.assertEqual(v.status,'IDLE')
        context=json.loads(actor_context(h,s,SimpleNamespace(geometry={}),allowed))['CURRENT preflight receipt']
        self.assertTrue(context['near_pose_gap']['fresh_execution_RGBD_recheck_required'])
        self.assertTrue(any(r.get('near_pose_gap_option') for r in context['scores_for_allowed_commands']))

    def test_extra_obstacle_veto_cannot_be_bypassed_by_original_depth_gate(self):
        _,s,h,_,c=setup();allowed=self.candidates(c,s,safe=False)
        extra=[r for r in h.candidate_receipt['tested'] if r.get('near_pose_gap_option')]
        self.assertTrue(extra)
        self.assertTrue(all(not r['accepted'] for r in extra))
        self.assertFalse(any(a.move in ROTATIONS for a in allowed))

    def test_default_and_available_improving_translation_do_not_get_extra_preflights(self):
        _,s,h,_,c=setup(False);self.candidates(c,s)
        self.assertNotIn('near_pose_gap',h.candidate_receipt)
        _,s,h,_,c=setup();c.candidates(s)
        self.assertEqual(h.candidate_receipt['near_pose_gap']['additional_preflights'],0)
        self.assertFalse(any(r.get('near_pose_gap_option') for r in h.candidate_receipt['tested']))

    def test_execution_requires_the_exact_accepted_option_and_actual_state(self):
        m,s,h,v,c=setup();allowed=self.candidates(c,s)
        action=next(a for a in allowed if a.move in ROTATIONS)
        self.assertTrue(execution_check(h,s,v.grips,m,v.limits,action)['eligible'])
        altered=copy.deepcopy(s);altered.q[11]+=.0001
        self.assertFalse(execution_check(h,altered,v.grips,m,v.limits,action)['eligible'])
        self.assertFalse(execution_check(h,s,[1,-1],m,v.limits,action)['eligible'])
        row=next(r for r in h.candidate_receipt['tested'] if r['action']==asdict(action))
        row['accepted']=False
        self.assertFalse(execution_check(h,s,v.grips,m,v.limits,action)['eligible'])
        row['accepted']=True;row['near_pose_sweep']['safe']=False
        self.assertFalse(execution_check(h,s,v.grips,m,v.limits,action)['eligible'])

    def test_new_execution_depth_and_robot_change_can_veto_a_prior_safe_option(self):
        m,s,h,v,c=setup();allowed=self.candidates(c,s)
        action=next(a for a in allowed if a.move in ROTATIONS)
        trial=SafeServo(m,s,v.grips.copy(),v.limits);self.assertTrue(trial.begin(action,s))
        marker={'fresh_sensor_not_candidate':True}
        for safe in (True,False):
            FakeSweep.safe=safe
            with patch('semantic_robot.v2.grounding.observed_cloud',return_value=np.ones((50,3))) as cloud, \
                 patch('semantic_robot.v2.arm_observation_guard.ObservingArmGuard',FakeSweep):
                ok,check=execution_sweep(h,s,s,v.grips,m,v.limits,action,marker,c.near_pose_geometry,trial.joint_plan)
                self.assertEqual(ok,safe);self.assertIs(cloud.call_args.args[0],marker)
                self.assertFalse(check['fresh_RGBD_sweep_still_required'])
        fresh=copy.deepcopy(s);fresh.gripper[0]-=.0001
        ok,check=execution_sweep(h,s,fresh,v.grips,m,v.limits,action,{},None,trial.joint_plan)
        self.assertFalse(ok);self.assertEqual(check['reason'],'NEAR_POSE_QUALIFICATION_CHANGED')

    def test_no_computation_after_deadline(self):
        _,s,_,_,c=setup()
        with self.assertRaises(WallTimeBudgetReached):c.candidates(s,deadline=0)


class NearPoseRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree=ast.parse((Path(__file__).resolve().parents[2]/'scripts/semantic_robot/run_v2.py').read_text())

    def test_default_off_dependencies_and_exact_gate(self):
        node=next(n for n in ast.walk(self.tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--near-pose-gap')
        parser=argparse.ArgumentParser();eval(compile(ast.Expression(node),'cli','eval'),{'p':parser})
        self.assertFalse(parser.parse_args([]).near_pose_gap);self.assertTrue(parser.parse_args(['--near-pose-gap']).near_pose_gap)
        node=next(n for n in ast.walk(self.tree) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('args.near_pose_gap and'))
        code=compile(ast.Module(body=[node],type_ignores=[]),'dependency','exec')
        for flags in range(16):
            enabled,rotate,geometry,review=[bool(flags&(1<<i)) for i in range(4)]
            args=SimpleNamespace(near_pose_gap=enabled,approach_reorientation=rotate,robot_geometry_guards=geometry,near_contact_review=review)
            if enabled and not (rotate and geometry and review):
                with self.assertRaises(ValueError):exec(code,{'args':args})
            else:exec(code,{'args':args})
        node=next(n for n in ast.walk(self.tree) if isinstance(n,ast.Compare) and ast.unparse(n.left)=="g.get('near_pose_gap', False)")
        code=compile(ast.Expression(node),'gate','eval')
        for gate in ({},{'near_pose_gap':False},{'near_pose_gap':True}):
            for requested in (False,True):
                self.assertEqual(eval(code,{'g':gate,'args':SimpleNamespace(near_pose_gap=requested)}),gate.get('near_pose_gap',False)==requested)

    def test_runner_captures_new_depth_before_sweep_and_veto_breaks(self):
        node=next(n for n in ast.walk(self.tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='accepted and near_option')
        calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)]
        capture=next(n for n in calls if ast.unparse(n.func)=='observation_now')
        sweep=next(n for n in calls if ast.unparse(n.func)=='execution_sweep')
        self.assertLess(capture.lineno,sweep.lineno)
        self.assertIn('fresh_depths',[ast.unparse(a) for a in sweep.args])
        veto=next(n for n in ast.walk(node) if isinstance(n,ast.If) and ast.unparse(n.test)=='not safe')
        self.assertTrue(any(isinstance(n,ast.Break) for n in veto.body))
        self.assertIn('accepted_before_motion=False',ast.unparse(veto))
        values=[v for n in ast.walk(self.tree) if isinstance(n,ast.Dict) for k,v in zip(n.keys,n.values)
                if isinstance(k,ast.Constant) and k.value=='near_pose_gap']
        self.assertEqual([ast.unparse(v) for v in values],['args.near_pose_gap']*2)


if __name__=='__main__':unittest.main()

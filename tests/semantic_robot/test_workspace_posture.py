"""H28 public qualification, bounded first-action preview, and runner identity."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action, TRANSLATIONS
from semantic_robot.v2.servo import SafeServo, ServoLimits
from semantic_robot.v2.wall_budget import WallTimeBudgetReached
from semantic_robot.v2.workspace_posture import qualification, preview, execution_check
from test_hand_body_collision import calibrated_fixture
from test_v2 import evidence


def setup():
    model, state, _ = calibrated_fixture()
    model.spec['metadata']['grasp_region_reference_gripper_m'] = [.05, .05]
    harness=GroundedHarness([Goal('pick','visible object','right','moves with hand')],workspace_posture=True)
    harness.stage='APPROACH';harness.observation=evidence()
    harness.last_gripper=state.gripper.copy()
    point=np.array([.7,-.2,1.])
    harness.grounding={'valid':True,'point_base_m':point.tolist(),
        'distance_to_active_closing_center_m':float(np.linalg.norm(point-model.grasp_centers(state.q)['right']))}
    servo=SafeServo(model,state,[1.,1.],limits=ServoLimits(robot_geometry_guards=True))
    controller=GroundedController(model,servo,harness)
    controller.target=harness.grounding;controller.centers=model.grasp_centers(state.q)
    controller.depth_guard=SimpleNamespace(check=lambda *a:(True,'FIXTURE_ONLY'),receipt=lambda:{})
    return model,state,harness,servo,controller


ORIGINAL_BEGIN=SafeServo.begin
def blocked_before_posture(trial, action, state, carry=False):
    if action.part=='right' and action.move in TRANSLATIONS and action.scale in ('fine','coarse') and np.max(abs(state.q[:3]))<1e-6:
        return trial.abort('UNREACHABLE_OR_COLLISION_BLOCKED')
    return ORIGINAL_BEGIN(trial,action,state,carry)


class WorkspacePostureTests(unittest.TestCase):
    def test_qualification_requires_exact_both_open_public_state(self):
        model,state,h,servo,_=setup()
        self.assertTrue(qualification(h,state,servo.grips,model)['eligible'])
        for key in ('pending_grasp','hold_verified','possible_contact_after_close','workspace_close_seen'):
            for invalid in (None,{}, {'left':False}, {'left':False,'right':None}, {'left':False,'right':0}, {'left':True,'right':False}):
                model,state,h,servo,_=setup();setattr(h,key,invalid)
                with self.subTest(key=key,invalid=invalid):
                    try:result=qualification(h,state,servo.grips,model)['eligible']
                    except (TypeError,AttributeError):continue  # malformed old carry contract also rejects
                    self.assertFalse(result)

    def test_missing_calibration_closed_latch_nonfinite_or_wrong_aperture_rejects(self):
        for mutate in (lambda m,s,h,g:m.spec['metadata'].pop('grasp_region_reference_gripper_m'),
                       lambda m,s,h,g:m.spec['metadata']['grasp_region_reference_fully_open'].update(left=False),
                       lambda m,s,h,g:s.gripper.__setitem__(0,.03),
                       lambda m,s,h,g:s.gripper.__setitem__(1,np.nan),
                       lambda m,s,h,g:g.__setitem__(0,-1),lambda m,s,h,g:g.__setitem__(1,1.01)):
            model,state,h,servo,_=setup();mutate(model,state,h,servo.grips)
            self.assertFalse(qualification(h,state,servo.grips,model)['eligible'])

    def test_close_then_open_does_not_clear_irreversible_workspace_latch(self):
        _,_,h,_,_=setup()
        h.executed(Action('left','close'),{'status':'INTERRUPTED','control_ticks':1,'finger_mean_m':[.05,.05]})
        h.executed(Action('left','open'),{'status':'TARGET_REACHED','control_ticks':18,'finger_mean_m':[.05,.05],
                                         'gripper_open_at_calibrated_aperture':['left','right']})
        self.assertTrue(h.workspace_close_seen['left'])
        self.assertFalse(h.possible_contact_after_close['left'])

    def test_stage_hazard_enclosure_target_or_goal_restrictions(self):
        changes=[lambda h:setattr(h,'stage','GRASP'),lambda h:setattr(h,'stage','SEARCH'),
                 lambda h:setattr(h,'stop_reason','STOP'),lambda h:setattr(h,'observation',evidence(hazard='collision')),
                 lambda h:setattr(h,'observation',evidence(enclosed=True)),lambda h:h.grounding.update(valid=False),
                 lambda h:h.grounding.update(distance_to_active_closing_center_m=np.nan),
                 lambda h:setattr(h,'goals',[Goal('pick','x','both','held')]),
                 lambda h:setattr(h,'goals',[Goal('press','x','right','effect')])]
        for mutate in changes:
            model,state,h,servo,_=setup();mutate(h)
            self.assertFalse(qualification(h,state,servo.grips,model)['eligible'])

    def test_existing_moderate_progress_does_not_trigger_more_body_checks(self):
        _,state,h,_,c=setup();c.candidates(state)
        receipt=h.candidate_receipt['workspace_posture']
        self.assertEqual(receipt['trigger'],'NO_MODERATE_WORKSPACE_BLOCKAGE')
        self.assertFalse(any(r['action']['part']=='torso' and r['action']['scale']=='fine' for r in h.candidate_receipt['tested']))

    def test_disabled_preserves_old_candidates_and_receipt(self):
        _,state,h,_,c=setup();h.workspace_posture=False
        with patch.object(SafeServo,'begin',blocked_before_posture):allowed=c.candidates(state)
        self.assertNotIn('workspace_posture',h.candidate_receipt)
        self.assertFalse(any(a.part=='torso' and a.scale=='fine' for a in allowed))

    def test_blocked_reach_exposes_only_useful_safe_first_actions(self):
        model,state,h,servo,c=setup();q=state.q.copy()
        with patch.object(SafeServo,'begin',blocked_before_posture):allowed=c.candidates(state)
        r=h.candidate_receipt;look=r['workspace_posture']['preview']
        self.assertTrue(look['future_actions_not_authorized'])
        self.assertLessEqual(look['additional_preflights'],12)
        self.assertEqual(len(look['rows']),4)
        self.assertTrue(any(a.part=='torso' and a.scale=='fine' for a in allowed))
        self.assertFalse(any(a.part=='right' and a.move in TRANSLATIONS and a.scale in ('fine','coarse') for a in allowed))
        self.assertLessEqual(len(r['tested'])-1,c.max_preflights+4)
        self.assertTrue(all(a in h.palette() for a in allowed))
        for row in r['tested']:
            if row['action']['part']=='torso' and row['action']['scale']=='fine':
                self.assertEqual(row['offered_to_policy'],Action(**row['action']) in allowed)
        np.testing.assert_array_equal(state.q,q);np.testing.assert_array_equal(servo.grips,[1,1])
        self.assertEqual(servo.status,'IDLE')
        displayed=json.loads(actor_context(h,state,SimpleNamespace(geometry={}),allowed))['CURRENT preflight receipt']
        self.assertTrue(displayed['workspace_posture']['future_actions_not_authorized'])
        self.assertNotIn('joint_plan',json.dumps(displayed))
        self.assertTrue(any('workspace_posture_after' in row for row in displayed['scores_for_allowed_commands']))

    def test_current_depth_rejection_cannot_be_overridden_by_prediction(self):
        _,state,h,_,c=setup()
        c.depth_guard.check=lambda a,*_:(a.part!='torso','FIXTURE_DEPTH_VETO')
        with patch.object(SafeServo,'begin',blocked_before_posture):allowed=c.candidates(state)
        self.assertFalse(any(a.part=='torso' for a in allowed))
        self.assertEqual(h.candidate_receipt['workspace_posture']['preview']['additional_preflights'],0)

    def test_no_feasible_future_never_exposes_a_body_command(self):
        _,state,h,_,c=setup()
        def blocked(trial,a,current,carry=False):
            if a.part=='right' and a.move in TRANSLATIONS and a.scale in ('fine','coarse'):
                return trial.abort('UNREACHABLE_OR_COLLISION_BLOCKED')
            return ORIGINAL_BEGIN(trial,a,current,carry)
        with patch.object(SafeServo,'begin',blocked):allowed=c.candidates(state)
        self.assertFalse(any(a.part=='torso' and a.scale=='fine' for a in allowed))
        self.assertTrue(all(r['summary']['feasible_followups']==0 for r in h.candidate_receipt['workspace_posture']['preview']['rows']))

    def test_guard_disabled_or_closed_history_never_qualifies(self):
        for change in ('guards','closed'):
            model,state,h,servo,c=setup()
            if change=='guards':c.servo=SafeServo(model,state,[1,1])
            else:h.workspace_close_seen['left']=True
            with patch.object(SafeServo,'begin',blocked_before_posture):allowed=c.candidates(state)
            self.assertFalse(h.candidate_receipt['workspace_posture']['eligible'])
            self.assertFalse(any(a.part=='torso' and a.scale=='fine' for a in allowed))

    def test_execution_rechecks_current_qualification_and_first_action_only(self):
        model,state,h,servo,c=setup()
        with patch.object(SafeServo,'begin',blocked_before_posture):allowed=c.candidates(state)
        action=next(a for a in allowed if a.part=='torso' and a.scale=='fine')
        self.assertTrue(execution_check(h,state,servo.grips,model,servo.limits,action)['eligible'])
        self.assertFalse(execution_check(h,state,servo.grips,model,servo.limits,Action('right','forward'))['eligible'])
        for change in ('closed','stage','aperture','unoffered'):
            changed=copy.deepcopy(h);current=copy.deepcopy(state)
            if change=='closed':changed.workspace_close_seen['left']=True
            elif change=='stage':changed.stage='ALIGN'
            elif change=='aperture':current.gripper[0]=.02
            else:
                for row in changed.candidate_receipt['tested']:row['offered_to_policy']=False
            self.assertFalse(execution_check(changed,current,servo.grips,model,servo.limits,action)['eligible'])

    def test_expired_deadline_prevents_even_first_depth_check(self):
        _,state,_,_,c=setup()
        with patch.object(c.depth_guard,'check') as call:
            with self.assertRaises(WallTimeBudgetReached):c.candidates(state,deadline=0)
            call.assert_not_called()

    def test_preview_exact_preflight_identity_and_boundaries(self):
        model,state,h,servo,_=setup();a=Action('torso','down','fine')
        trial=SafeServo(model,state,servo.grips.copy(),servo.limits);self.assertTrue(trial.begin(a,state))
        following=[Action('right','forward')]
        def call(pairs=((a,trial),),next_actions=following):
            return preview(model,state,servo.grips,servo.limits,'right',np.array(h.grounding['point_base_m']),pairs,next_actions)
        self.assertLessEqual(call()['additional_preflights'],3)
        for pairs,next_actions in (([(a,trial)]*5,following), ([(a,trial)],following*4),
                                  ([(a,trial)],[Action('left','forward')]),
                                  ([(Action('torso','up'),trial)],following)):
            with self.assertRaises(ValueError):call(pairs,next_actions)
        trial.grips[0]=-1
        with self.assertRaises(ValueError):call()

    def test_runner_exact_flag_identity_and_guard(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/'scripts/semantic_robot/run_v2.py').read_text())
        comparison=next(n for n in ast.walk(tree) if isinstance(n,ast.Compare) and ast.unparse(n.left)=="g.get('workspace_posture', False)")
        code=compile(ast.Expression(body=comparison),'actual_workspace_gate','eval')
        for gate,requested,expected in (({},False,True),({},True,False),({'workspace_posture':True},True,True),({'workspace_posture':True},False,False)):
            self.assertEqual(eval(code,{'g':gate,'args':SimpleNamespace(workspace_posture=requested)}),expected)
        values=[v for n in ast.walk(tree) if isinstance(n,ast.Dict) for k,v in zip(n.keys,n.values)
                if isinstance(k,ast.Constant) and k.value=='workspace_posture']
        self.assertEqual(len(values),2)
        self.assertTrue(all(ast.unparse(v)=='args.workspace_posture' for v in values))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                  and ast.unparse(n.test)=="args.workspace_posture and (not args.robot_geometry_guards)")
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_workspace_guard','exec')
        with self.assertRaises(ValueError):exec(code,{'args':SimpleNamespace(workspace_posture=True,robot_geometry_guards=False)})
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                  and ast.unparse(n.test)=="args.workspace_posture and (args.prefix or args.replay_prefix_spec)")
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_workspace_start','exec')
        for prefix,replay in ((1,None),(0,'saved')):
            with self.assertRaises(ValueError):exec(code,{'args':SimpleNamespace(workspace_posture=True,prefix=prefix,replay_prefix_spec=replay)})

    def test_real_gate_profile_exercises_fine_torso_without_extra_actions(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/'scripts/semantic_robot/run_v2.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bounded_gate')
        code=compile(ast.Module(body=[node],type_ignores=[]),'actual_gate_action_builder','exec')
        from semantic_robot.v2.protocol import HOLD
        scope={'Action':Action,'HOLD':HOLD};exec(code,scope)
        original=scope['bounded_gate'](True,True,False);changed=scope['bounded_gate'](True,True,True)
        self.assertEqual(len(original),24);self.assertEqual(len(changed),24)
        self.assertEqual([i for i,(a,b) in enumerate(zip(original,changed)) if a!=b],[9,10])
        self.assertEqual(changed[9:11],[Action('torso','up'),Action('torso','down')])

    def test_actual_runner_recheck_stops_before_changed_qualification_motion(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/'scripts/semantic_robot/run_v2.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
                  ast.unparse(n.test)=="args.workspace_posture and manager and (action.part == 'torso') and (action.scale == 'fine')")
        loop=ast.parse('for _ in range(1):\n    pass').body[0];loop.body=[node]
        code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'actual_workspace_recheck','exec')
        for changed in (False,True):
            model,state,h,servo,c=setup()
            with patch.object(SafeServo,'begin',blocked_before_posture):allowed=c.candidates(state)
            action=next(a for a in allowed if a.part=='torso' and a.scale=='fine')
            h.workspace_close_seen['left']=changed
            writes=[];scope={'args':SimpleNamespace(workspace_posture=True),'manager':h,'state':state,'servo':servo,
                'action':action,'model':model,'directory':Path('/unused_cpu_fixture'),
                'write':lambda p,v:writes.append(v),'row':{},'decisions':[]}
            exec(code,scope)
            self.assertEqual(writes[0]['eligible'],not changed)
            self.assertEqual(len(scope['decisions']),int(changed))
            self.assertEqual(servo.status,'IDLE')
            if changed:self.assertEqual(h.stop_reason,'WORKSPACE_POSTURE_QUALIFICATION_CHANGED')


if __name__=='__main__':unittest.main()

"""Explicit body2/carry timing, actual issuance and all paired actor identities."""
import ast
import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft'),str(ROOT/'tests/semantic_robot')]
from common import sha,write_json
from native_execution import (CARRY_PROFILE,WORKSPACE_PROFILE,metadata,authorization_profile,episode_limits,
    primitive_tick_limit,servo_limits,service_call_limit,WorkspaceQuota,PublicGripperHistory,body_limit,
    require_dataset_profile,require_pipeline_profile,require_same_pipeline,completed,workspace_budget_ok,actor_protocol)
from native_motion_codec import BODY_TOKENS,VERSION as CODEC,token_to_action
from native_evaluation import PublicExecution,VARIANTS
from native_teacher_collect import native_preflight
from native_teacher_workspace import fallback
from native_dataset import check_actual_workspace,check_public_timing
from native_eval_run import evaluation_budget,require_evaluation,BUDGET
from native_carry_admission import (INHERITANCE,BASE_EXECUTOR,BOOTSTRAP_NAME,require_carry_reviews,require_engineering_gates)
import native_teacher_capacity as capacity
import native_storage as storage
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import native_action
from test_native_preclose_translation import fixture,authorization
from test_native_workspace import capture


def auth():
    return {**authorization(),**metadata(CARRY_PROFILE),'capacity_profile':capacity.CARRY_PROFILE,
        'storage':copy.deepcopy(storage.CARRY_SPEC),'protocol':actor_protocol(CARRY_PROFILE),
        'paid_prefix_controls':388,'authorize_offline_teacher':True,'executor_digest':'a'*64,'code_commit':'d'*40,
        'carry_duration_core_commit':'b'*40,'carry_duration_bootstrap':True,
        'engineering_gate_inheritance':copy.deepcopy(INHERITANCE),'carry_duration_integration_review':None}


def receipts(root,value,*,integrated=False):
    cpu={'reviewer':'Codex-parent','review_passed':True,'executor_digest':value['executor_digest'],
        'core_commit':value['carry_duration_core_commit'],'carry_duration_v1':True,
        'original_safety_and_rate_limits_unchanged':True}
    p=root/'cpu.json';write_json(p,cpu);value['carry_duration_cpu_review']={'path':str(p),'sha256':sha(p)}
    if integrated:
        p=root/'integration.json';write_json(p,{**metadata(CARRY_PROFILE),'reviewer':'Codex-parent','review_passed':True,
            'executor_digest':value['executor_digest'],'core_commit':value['carry_duration_core_commit'],
            'run_name':BOOTSTRAP_NAME,'integration_passed':True,'completed_extended_carry_macro_ticks':[44],
            'inventory_sha256':'c'*64})
        value['carry_duration_bootstrap']=False;value['carry_duration_integration_review']={'path':str(p),'sha256':sha(p)}
    return value


class CarryNativeTests(unittest.TestCase):
    def test_exact_contract_old_stays_unchanged(self):
        a=auth();self.assertEqual(authorization_profile(a,collection=True),CARRY_PROFILE)
        self.assertEqual(episode_limits(CARRY_PROFILE),(14,640));self.assertEqual(episode_limits(WORKSPACE_PROFILE),(12,420))
        self.assertEqual(service_call_limit(CARRY_PROFILE),56);self.assertEqual(service_call_limit(WORKSPACE_PROFILE),48)
        self.assertEqual(body_limit(CARRY_PROFILE),2);self.assertEqual(body_limit(WORKSPACE_PROFILE),1)
        for key,value in (('carry_duration_v1',1),('max_workspace_macros',3),('max_workspace_macros',True),
                          ('max_decisions',12),('new_controls_including_final_hold',420),('action_codec',None)):
            with self.subTest(key=key),self.assertRaises(ValueError):authorization_profile({**a,key:value},collection=True)
        with self.assertRaises(ValueError):authorization_profile({'carry_duration_v1':True})
        with self.assertRaises(ValueError):authorization_profile({**a,'execution_profile':WORKSPACE_PROFILE})
        self.assertEqual(evaluation_budget(CARRY_PROFILE),{**BUDGET,'max_decisions':14,'new_controls_including_final_hold':640})

    def test_exact_two_new_paths_and_no_old_alias_authority(self):
        for prefix in (388,380):
            a={**auth(),'paid_prefix_controls':prefix};name=f'native_t1_i192_p{prefix:04d}_ws45_cd1_b2'
            capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/name,3)
            storage.validate_spec(a)
            self.assertIn(name,storage.CARRY_RUNS);self.assertNotIn(name,storage.WORKSPACE_RUNS)
            for wrong in (name+'_retry',name.replace('_cd1_b2','')):
                with self.assertRaises(ValueError):capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/wrong,3)
        for prefix in (392,396,389):
            with self.assertRaises(ValueError):capacity.capacity_limits({**auth(),'paid_prefix_controls':prefix})
        with self.assertRaises(ValueError):authorization_profile({**auth(),'storage':storage.WORKSPACE_SPEC},collection=True)
        with self.assertRaises(ValueError):authorization_profile({**auth(),'capacity_profile':capacity.WORKSPACE_PROFILE},collection=True)

    def test_shared_quota_once_per_actual_macro_not_per_tick(self):
        q=WorkspaceQuota(CARRY_PROFILE);command=np.zeros(23);command[[14,22]]=1
        q.arm(BODY_TOKENS[0]);self.assertEqual(q.count,0)
        q.cancel_pending();q.issued(command);self.assertEqual(q.count,0)
        for count in (1,2):
            q.arm(BODY_TOKENS[1])
            for _ in range(40):q.issued(command)
            self.assertEqual(q.count,count)
            q.arm('RIGHT_OPEN');q.issued(command);self.assertEqual(q.count,count)
        with self.assertRaises(RuntimeError):q.arm(BODY_TOKENS[1])
        q=WorkspaceQuota(CARRY_PROFILE);q.arm(BODY_TOKENS[1])
        for bad in (np.zeros(22),np.full(23,np.nan),np.full(23,2)):
            with self.assertRaises(ValueError):q.issued(bad)
            self.assertEqual(q.count,0)

    def test_collector_public_all_three_variants_same_profile_and_body2(self):
        model,state,geometry=fixture();depths,clock,receipt=capture(model,state,geometry)
        cloud=np.tile([1.5,0,.2],(50,1));token=BODY_TOKENS[1]
        with patch('native_teacher_collect.observed_cloud',return_value=cloud):
            teacher,tc=native_preflight(model,state,[1,1],token_to_action(token,CODEC),depths,geometry,
                execution_profile=CARRY_PROFILE,public_history=PublicGripperHistory([1,1]))
        for variant in VARIANTS:
            policy=PublicExecution([1,1],execution_profile=CARRY_PROFILE)
            for count in (1,2):
                with patch('native_evaluation.observed_cloud',return_value=cloud):
                    servo,pc=policy.preflight(token,state,model,depths,geometry,capture_receipt=receipt,expected_clock=clock)
                np.testing.assert_array_equal(servo.joint_plan,teacher.joint_plan)
                self.assertEqual(servo.limits,teacher.limits,variant);self.assertTrue(servo.limits.carry_duration_v1)
                self.assertEqual(authorization_profile(tc),authorization_profile(pc))
                for _ in range(3):policy.issued(native_action(state.q,[1,1]))
                self.assertEqual(policy.workspace_quota.count,count)
            with self.assertRaises(RuntimeError):policy.preflight(token,state,model,depths,geometry,capture_receipt=receipt,expected_clock=clock)
        policy=PublicExecution([1,1],execution_profile=CARRY_PROFILE);policy.issued(native_action(state.q,[1,-1]));policy.issued(native_action(state.q,[1,1]))
        with self.assertRaises(RuntimeError):policy.preflight(token,state,model,depths,geometry,capture_receipt=receipt,expected_clock=clock)

    def test_new_timing_receipt_strict_and_legacy_rejects_new(self):
        model,state,_=fixture();token='RIGHT_UP';servo=__import__('semantic_robot.v2.servo',fromlist=['SafeServo']).SafeServo(model,state,
            gripper_command=[1,-1],limits=servo_limits(CARRY_PROFILE))
        self.assertTrue(servo.begin(token_to_action(token),state,carry=True))
        while not servo.done and servo.ticks<servo.total_ticks:
            c=servo.next_action(state);state=model.state(np.r_[c[3:14],c[15:22]],state.gripper,np.zeros(3))
        f=servo.finish(state);self.assertTrue(completed(token,f,profile=CARRY_PROFILE))
        self.assertFalse(completed(token,f,profile=WORKSPACE_PROFILE))
        for field,value in (('maximum_control_ticks',40),('joint_step_limit',.025),('success_claim',True),
                            ('required_joint_trajectory_ticks',1),('joint_trajectory_planned',1),('joint_trajectory_settling_ticks',6)):
            bad=copy.deepcopy(f);bad['motion_timing'][field]=value
            self.assertFalse(completed(token,bad,profile=CARRY_PROFILE),field)
        bad=copy.deepcopy(f);bad.pop('motion_timing');self.assertFalse(completed(token,bad,profile=CARRY_PROFILE))
        for action,carry,limit in ((Action('right','up'),True,75),(Action('both','up'),True,75),
            (Action('torso','up'),True,40),(Action('base','forward'),True,40),(Action('right','up'),False,40),
            (Action('right','close'),True,40)):
            self.assertEqual(primitive_tick_limit(action,carry,CARRY_PROFILE),limit)

    def test_exact_worst_case_body_reservation_stays_inside_episode(self):
        # 18 body +12 settle+75 next+12 settle+1 hold=118, still two macro slots.
        self.assertFalse(workspace_budget_ok(BODY_TOKENS[0],18,117,2,CARRY_PROFILE))
        self.assertTrue(workspace_budget_ok(BODY_TOKENS[0],18,118,2,CARRY_PROFILE))
        self.assertFalse(workspace_budget_ok(BODY_TOKENS[0],18,640,1,CARRY_PROFILE))
        self.assertTrue(workspace_budget_ok(BODY_TOKENS[0],18,83,2,WORKSPACE_PROFILE))

    def test_dataset_replays_new_motion_and_gripper_receipts_from_each_actual_q(self):
        from semantic_robot.v2.servo import SafeServo
        for token in ('RIGHT_UP','RIGHT_CLOSE'):
            model,state,_=fixture();initial=state
            before={'q':state.q.tolist(),'gripper':state.gripper.tolist()}
            history=PublicGripperHistory([1,-1]);commands=[{'action23':native_action(state.q,history.grips).tolist()}]
            servo=SafeServo(model,state,gripper_command=history.grips,limits=servo_limits(CARRY_PROFILE))
            self.assertTrue(servo.begin(token_to_action(token),state,carry=True));telemetry={}
            for i in range(1,servo.total_ticks+1):
                cmd=servo.next_action(state);commands.append({'action23':cmd.tolist()})
                state=model.state(np.r_[cmd[3:14],cmd[15:22]],state.gripper,np.zeros(3))
                telemetry[1+i]={'actual_q':state.q.tolist(),'actual_gripper':state.gripper.tolist()}
            feedback=servo.finish(state);after={'q':state.q.tolist(),'gripper':state.gripper.tolist()}
            execution={'control_start':0,'control_end':servo.total_ticks,'feedback':feedback}
            check_actual_workspace(token,before,after,execution,model,commands,telemetry,1,history,CARRY_PROFILE)
            altered=copy.deepcopy(execution);altered['feedback']['motion_timing']['maximum_control_ticks']=999
            with self.assertRaises(ValueError):check_actual_workspace(token,before,after,altered,model,commands,telemetry,1,history,CARRY_PROFILE)
            changed=copy.deepcopy(commands);changed[1]['action23'][7]+=.0001
            with self.assertRaises(ValueError):check_actual_workspace(token,before,after,execution,model,changed,telemetry,1,history,CARRY_PROFILE)
            if token.endswith('CLOSE'):
                altered=copy.deepcopy(execution);altered['feedback']['gripper_execution']['success_claim']=True
                with self.assertRaises(ValueError):check_actual_workspace(token,before,after,altered,model,commands,telemetry,1,history,CARRY_PROFILE)

    def test_bootstrap_cannot_self_certify_integration_or_train(self):
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),auth());require_carry_reviews(a,bootstrap=True)
            with self.assertRaises(ValueError):require_carry_reviews(a)
            with self.assertRaises(ValueError):require_carry_reviews({**a,'paid_prefix_controls':380},bootstrap=True)
            with self.assertRaises(ValueError):require_carry_reviews({**a,'carry_duration_bootstrap':1},bootstrap=True)
            data={'protocol':actor_protocol(CARRY_PROFILE),'runs':[{},metadata(WORKSPACE_PROFILE),metadata(CARRY_PROFILE)]}
            with self.assertRaises(ValueError):require_pipeline_profile(a,data)
            a=receipts(Path(d),a,integrated=True);require_carry_reviews(a)
            self.assertEqual(require_pipeline_profile(a,data),CARRY_PROFILE)
            require_same_pipeline(a,a)
            for key,value in (('carry_duration_v1',False),('max_decisions',12),('storage',storage.WORKSPACE_SPEC)):
                with self.assertRaises(ValueError):require_same_pipeline(a,{**a,key:value})
            with self.assertRaises(ValueError):require_dataset_profile(data,WORKSPACE_PROFILE)
            p=Path(a['carry_duration_integration_review']['path']);r=json.loads(p.read_text());r['completed_extended_carry_macro_ticks']=[40]
            write_json(p,r);a['carry_duration_integration_review']['sha256']=sha(p)
            with self.assertRaises(ValueError):require_carry_reviews(a)

    def test_old_gate_inheritance_is_exact_partial_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=receipts(root,auth());paths=[];hashes={}
            for task in (0,3):
                p=root/f'g{task}.json';write_json(p,{'task':task,'gate_ok':True,'robot_geometry_guards':True,
                    'gripper_completion_v1':True,'implementation_digest':BASE_EXECUTOR})
                paths.append(str(p));hashes[task]=sha(p)
            a['engineering_gate_paths']=paths
            with patch('native_carry_admission.BASE_GATES',hashes):
                require_engineering_gates(a,a['executor_digest'],bootstrap=True)
                with self.assertRaises(ValueError):require_engineering_gates({**a,'engineering_gate_inheritance':{}},a['executor_digest'],bootstrap=True)
                p=Path(paths[0]);r=json.loads(p.read_text());r['carry_duration_v1']=True;write_json(p,r)
                with self.assertRaises(ValueError):require_engineering_gates(a,a['executor_digest'],bootstrap=True)

    def test_all_six_evaluation_authorizations_use_new_same_budget(self):
        from native_eval_prepare import ROOT as EXPERIMENT,SCHEMA
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),auth(),integrated=True)
            for k in ('capacity_profile','seed_profile','authorize_offline_teacher'):a.pop(k)
            a.update(schema=SCHEMA,authorize_evaluation=True,code_commit='code',dataset_sha256='data',physical_gpu=3,
                specified_hand=None,oracle_actor_feedback=False,reviewer='Codex-parent',budget=evaluation_budget(CARRY_PROFILE))
            with patch('native_carry_admission.require_engineering_gates'):
                for instance,episode in ((1,200),(71,247)):
                    for variant in VARIANTS:
                        a.update(source=[1,episode,instance],variant=variant)
                        output=EXPERIMENT/f'eval_t1_i{instance}_{variant}_v1'
                        require_evaluation(a,'code',a['executor_digest'],{'source':a['source']},variant,output,'data')
                        with self.assertRaises(ValueError):require_evaluation({**a,'budget':BUDGET},'code',a['executor_digest'],{'source':a['source']},variant,output,'data')

    def test_actual_collector_and_eval_issue_boundary_count_partial_not_enter_failure(self):
        class Globalize(ast.NodeTransformer):
            def visit_Nonlocal(self,node):return ast.Global(names=node.names)
        class BadEnter:
            def __enter__(self):raise RuntimeError('enter failure')
            def __exit__(self,*args):pass
        for filename in ('native_teacher_collect.py','native_eval_run.py'):
            tree=ast.parse((ROOT/'scripts/vlm_sft'/filename).read_text())
            fn=Globalize().visit(copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')))
            command=np.zeros(23,np.float32);command[[14,22]]=1
            quota=WorkspaceQuota(CARRY_PROFILE);quota.arm(BODY_TOKENS[1])
            policy=PublicExecution([1,1],execution_profile=CARRY_PROFILE);policy.workspace_quota.arm(BODY_TOKENS[1])
            issue=io.StringIO();done=io.StringIO();state=NS(q=np.zeros(18),gripper=np.ones(2)*.05)
            writer=NS(check=lambda *a:None,append_text=lambda s,x:s.write(x))
            def fail(*a,**k):raise RuntimeError('env failure')
            ns={'np':np,'json':json,'execution_profile':CARRY_PROFILE,'CARRY_PROFILE':CARRY_PROFILE,'near':True,
                'native_limit':640,'controls':12,'prefix_count':388,'issued_native':12,'issued_prefix':388,'terminal':False,
                'grips':np.ones(2),'teacher_frame':None,'teacher_reader':None,'public_history':PublicGripperHistory([1,1]),
                'workspace_quota':quota,'policy':policy,'last_info':{},'reader':None,'started':0,'wall_limit':1200,
                'budget':{'seconds_after_reset':1200},
                'time':NS(monotonic=lambda:1),'storage':None,'check_storage':lambda *a:None,
                'check_actual_joint_bounds':lambda *a:None,'state':lambda:state,'model':object(),
                'shutil':NS(disk_usage=lambda *a:NS(free=100*1024**3)),'x':NS(output=Path('/unused')),
                'current_writer':writer,'issue_trace':issue,'trace':done,'og':NS(sim=NS(render_on_step=lambda _:BadEnter())),
                'env':NS(step=fail)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),filename,'exec'),ns)
            with self.assertRaisesRegex(RuntimeError,'enter'):ns['step'](command,'candidate')
            active=quota if filename=='native_teacher_collect.py' else policy.workspace_quota
            self.assertEqual(active.count,0);self.assertEqual(ns['issued_native'],12);self.assertEqual(issue.getvalue(),'')
            ns['og']=NS(sim=NS(render_on_step=lambda _:nullcontext()))
            with self.assertRaisesRegex(RuntimeError,'env'):ns['step'](command,'candidate')
            self.assertEqual(active.count,1);self.assertEqual(ns['issued_native'],13);self.assertEqual(ns['controls'],12)
            self.assertEqual(done.getvalue(),'');self.assertEqual(len(issue.getvalue().splitlines()),1)


if __name__=='__main__':unittest.main()

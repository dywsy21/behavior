"""Explicit native45, actual-command eligibility and bounded PRIVATE lookahead."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft'),str(ROOT/'tests/semantic_robot')]
import common
import native_motion_codec as codec
import native_actor_protocol as protocol
from native_execution import (WORKSPACE_PROFILE,PRECLOSE_PROFILE,PROFILE,metadata,action_codec,actor_protocol,
    PublicGripperHistory,workspace_qualified,workspace_budget_ok,completed,authorization_profile,
    require_pipeline_profile,require_same_pipeline)
from native_evaluation import PublicExecution,VARIANTS,NearestNeighbor,choose,features
from native_teacher_collect import native_preflight
from native_teacher_workspace import fallback
from native_teacher_artifacts import json_bytes
from native_dataset import check_public_timing,check_actual_workspace
from native_train import CONFIG,WORKSPACE_CONFIG,validate_config
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import native_action
from test_native_preclose_translation import fixture,authorization
import test_native_actor_protocol as actor_fixtures
import native_storage as storage
import native_teacher_capacity as capacity


def capture(model,state,geometry):
    depths={v:np.ones((2,2),np.float32) for v in common.CAMERAS}
    clock={'prefix_control':832,'native_control':12}
    receipt={'clock':clock,'kinematic_model_sha256':model.sha,'q':state.q.tolist(),
        'gripper':state.gripper.tolist(),'scene_truth':False,
        'files_sha256':{'robot_self_geometry.json':hashlib.sha256(json_bytes(geometry)).hexdigest()},
        'depth_array_sha256':{v:hashlib.sha256(d.tobytes()).hexdigest() for v,d in depths.items()}}
    return depths,clock,receipt


class WorkspaceTests(unittest.TestCase):
    def test_native45_is_explicit_and_old41_expert_semantics_unchanged(self):
        self.assertEqual(len(common.TOKENS),41);self.assertEqual(common.VERSION,'h09-semantic-motion-v4-no-torso')
        self.assertEqual(len(codec.tokens(codec.VERSION)),45);self.assertEqual(codec.tokens(),common.TOKENS)
        self.assertEqual(codec.tokens(codec.VERSION)[:41],common.TOKENS)
        for token in codec.BODY_TOKENS:
            with self.assertRaises(ValueError):common.token_to_action(token)
            with self.assertRaises(ValueError):codec.token_to_action(token)
            action=codec.token_to_action(token,codec.VERSION)
            self.assertEqual((action.part,action.scale,action.frame),('torso','fine','base'))
            self.assertEqual(action.amount(False),.01)
            self.assertFalse(token.endswith('_UP'))
        for token in ('TORSO_UP','TORSO_DOWN','TORSO_UP_COARSE','TORSO_LEFT_KEEP_EEF'):
            with self.assertRaises(ValueError):codec.token_to_action(token,codec.VERSION)
        with self.assertRaises(ValueError):codec.tokens('unknown')

    def test_same_new_actor_prefix_for_train_base_finetuned_and_nn(self):
        _,_,_,old,raw=actor_fixtures.PoseProtocolTests().fixture()
        new=protocol.actor_input(old['task'],old['active_instruction'],old['proprio'],old['current_rgb_sha256'],
            [codec.BODY_TOKENS[1]],protocol=codec.ACTOR_VERSION)
        train=protocol.training_row(new,'RIGHT_DOWN')
        from modeling import messages
        for variant in VARIANTS[:2]:
            row,images=protocol.parse_request(protocol.request_payload(new,raw,variant),registered_instructions=[new['active_instruction']])
            self.assertEqual(messages(train,images),messages(row,images))
            self.assertIn('compensating both arms',messages(row,images)[0]['content'])
            result=choose(variant,new,raw,remote=lambda _: {'prediction':codec.BODY_TOKENS[0],'variant':variant,'protocol':codec.ACTOR_VERSION})
            self.assertIn(result['prediction'],codec.BODY_TOKENS)
        nn=NearestNeighbor([{**train,'id':'one'}])
        self.assertEqual(choose('proprio_history_nn',new,raw,nn=nn)['prediction'],'RIGHT_DOWN')
        self.assertEqual(len(features(new))-len(features(old)),20)
        with self.assertRaises(ValueError):nn.predict(old)
        with self.assertRaises(ValueError):protocol.validate_actor({**new,'protocol':protocol.VERSION})
        for field in ('goal_pose','held','teacher','private_prediction','future_q'):
            with self.assertRaises(ValueError):protocol.validate_actor({**new,field:True})
        with self.assertRaises(ValueError):choose('base',new,raw,remote=lambda _:{'prediction':'RIGHT_UP','variant':'base','protocol':protocol.VERSION})

    def test_both_real_open_and_never_issued_close_required(self):
        model,state,_=fixture();action=codec.token_to_action(codec.BODY_TOKENS[1],codec.VERSION)
        history=PublicGripperHistory([1,1])
        self.assertTrue(workspace_qualified(action,state,model,history,WORKSPACE_PROFILE))
        for old in (None,PROFILE,PRECLOSE_PROFILE):self.assertFalse(workspace_qualified(action,state,model,history,old))
        for arm in (0,1):
            h=PublicGripperHistory([1,1]);c=native_action(state.q,[1,1]);c[[14,22][arm]]=-1;h.issued(c)
            h.issued(native_action(state.q,[1,1]))
            self.assertFalse(workspace_qualified(action,state,model,h,WORKSPACE_PROFILE))
            state.gripper[arm]=.049
            self.assertFalse(workspace_qualified(action,state,model,history,WORKSPACE_PROFILE));state.gripper[arm]=.05
        with self.assertRaises(ValueError):Action('torso','down',frame='tool')
        for other in (Action('torso','down','micro'),Action('torso','down','coarse'),Action('both','down')):
            self.assertFalse(workspace_qualified(other,state,model,history,WORKSPACE_PROFILE))
        model.spec['metadata']['grasp_region_reference_fully_open']['left']=1
        self.assertFalse(workspace_qualified(action,state,model,history,WORKSPACE_PROFILE))

    def test_real_collector_public_all_variants_and_single_issued_body_latch(self):
        model,state,geometry=fixture();depths,clock,receipt=capture(model,state,geometry)
        cloud=np.tile([1.5,0,.2],(50,1));token=codec.BODY_TOKENS[1]
        with patch('native_teacher_collect.observed_cloud',return_value=cloud):
            teacher,tc=native_preflight(model,state,[1,1],codec.token_to_action(token,codec.VERSION),depths,geometry,
                execution_profile=WORKSPACE_PROFILE,public_history=PublicGripperHistory([1,1]))
        for variant in VARIANTS:
            policy=PublicExecution([1,1],execution_profile=WORKSPACE_PROFILE)
            with patch('native_evaluation.observed_cloud',return_value=cloud):
                servo,pc=policy.preflight(token,state,model,depths,geometry,capture_receipt=receipt,expected_clock=clock)
            self.assertEqual(servo.total_ticks,teacher.total_ticks,variant)
            np.testing.assert_array_equal(servo.joint_plan,teacher.joint_plan)
            self.assertIs(pc['carry'],False);self.assertIs(pc['scene_sweep_clearance_certified'],False)
            self.assertEqual(pc['amount'],.01);self.assertEqual(pc['public_both_open_no_close_workspace_qualification'],tc['public_both_open_no_close_workspace_qualification'])
            self.assertFalse(policy.workspace_seen)
            policy.issued(servo.next_action(state));self.assertTrue(policy.workspace_seen)
            with patch('native_evaluation.observed_cloud',return_value=cloud),self.assertRaises(RuntimeError):
                policy.preflight(token,state,model,depths,geometry,capture_receipt=receipt,expected_clock=clock)
        stale=copy.deepcopy(receipt);stale['q'][0]+=.0001
        with self.assertRaises(ValueError):PublicExecution([1,1],execution_profile=WORKSPACE_PROFILE).preflight(token,state,model,depths,geometry,capture_receipt=stale,expected_clock=clock)

    def test_full_body_execution_and_dataset_replay_not_status_whitelist(self):
        model,state,geometry=fixture();token=codec.BODY_TOKENS[1];history=PublicGripperHistory([1,1])
        cloud=np.tile([1.5,0,.2],(50,1))
        with patch('native_teacher_collect.observed_cloud',return_value=cloud):
            servo,receipt=native_preflight(model,state,history.grips,codec.token_to_action(token,codec.VERSION),{},geometry,
                execution_profile=WORKSPACE_PROFILE,public_history=history)
        before={'q':state.q.tolist(),'gripper':state.gripper.tolist()};commands=[{'action23':native_action(state.q,[1,1]).tolist()}]
        telemetry={};current=state
        for control in range(1,servo.total_ticks+1):
            command=servo.next_action(current);commands.append({'action23':command.tolist()})
            current=model.state(np.r_[command[3:14],command[15:22]],state.gripper,np.zeros(3))
            telemetry[1+control]={'actual_q':current.q.tolist(),'actual_gripper':current.gripper.tolist()}
        feedback=servo.finish(current);self.assertTrue(completed(token,feedback,profile=WORKSPACE_PROFILE))
        execution={'control_start':0,'control_end':servo.total_ticks,'feedback':feedback}
        after={'q':current.q.tolist(),'gripper':current.gripper.tolist()}
        check_public_timing({'preflight':receipt},token,before,model,commands,1,0,WORKSPACE_PROFILE)
        check_actual_workspace(token,before,after,execution,model,commands,telemetry,1,history)
        for bad in ({'status':'TARGET_REACHED'},{**feedback,'carry':True},{**feedback,'control_ticks':1}):
            self.assertFalse(completed(token,bad,profile=WORKSPACE_PROFILE))
        corrupted=copy.deepcopy(commands);corrupted[2]['action23'][3]+=.001
        with self.assertRaises(ValueError):check_actual_workspace(token,before,after,execution,model,corrupted,telemetry,1,history)

    def test_bounded_private_fallback_and_no_prediction_to_actor_or_control(self):
        model,state,geometry=fixture();history=PublicGripperHistory([1,1]);cloud=np.tile([1.5,0,.2],(50,1))
        frame={'contacts_known':True,**{k:{a:False for a in ('left','right')} for k in ('held','held_any','finger_contact','finger_external_contact')}}
        goal=model.forward(state.q,'right').copy();goal[2,3]-=.03
        def preflight(token):
            with patch('native_teacher_collect.observed_cloud',return_value=cloud):
                return native_preflight(model,state,[1,1],codec.token_to_action(token,codec.VERSION),{},geometry,
                    execution_profile=WORKSPACE_PROFILE,public_history=history)
        kwargs=dict(model=model,state=state,frame=frame,goal_world=goal,base_world=np.eye(4),hand='right',history=history,
            ranked=['RIGHT_DOWN','RIGHT_FORWARD','RIGHT_PITCH_PLUS'],rejected=[{'token':t,'reason':'original veto'} for t in ('RIGHT_DOWN','RIGHT_FORWARD','RIGHT_PITCH_PLUS')],
            profile=WORKSPACE_PROFILE,used=False,remaining_controls=408,remaining_macros=12,preflight=preflight)
        choice,report=fallback(**kwargs)
        self.assertIsNotNone(choice);self.assertEqual(len(report['body_trials']),4)
        self.assertLessEqual(sum(len(r['following']) for r in report['body_trials']),12)
        self.assertEqual(choice[1].ticks,0);self.assertTrue(report['predicted_not_executed'])
        for changes in ({'used':True},{'remaining_controls':30},{'remaining_macros':1},{'ranked':[],'rejected':[]},{'rejected':[]}):
            self.assertIsNone(fallback(**{**kwargs,**changes})[0])
        bad=copy.deepcopy(frame);bad['finger_external_contact']['left']=True
        self.assertIsNone(fallback(**{**kwargs,'frame':bad})[0])
        def reject(_):raise RuntimeError('real safety veto')
        self.assertIsNone(fallback(**{**kwargs,'preflight':reject})[0])
        source=(ROOT/'scripts/vlm_sft/native_teacher_workspace.py').read_text()
        self.assertNotIn('env.step',source);self.assertNotIn('LocalOutcome',source);self.assertNotIn('observed_cloud',source)

    def test_exact_shared_budget_and_failclosed_cpu_registration(self):
        token=codec.BODY_TOKENS[0]
        self.assertTrue(workspace_budget_ok(token,26,91,2))
        for controls,macros in ((90,2),(100,1),(True,2)):
            self.assertFalse(workspace_budget_ok(token,26,controls,macros))
        self.assertTrue(workspace_budget_ok('RIGHT_UP',18,1,1))
        self.assertEqual(capacity.WORKSPACE_STARTS,frozenset())
        self.assertEqual(storage.WORKSPACE_RUNS,storage.DIVERSE_RUNS)
        a=authorization();a.update(**metadata(WORKSPACE_PROFILE),capacity_profile=capacity.WORKSPACE_PROFILE,storage=storage.WORKSPACE_SPEC)
        with self.assertRaises(ValueError):authorization_profile(a,collection=True)
        for cfg in (CONFIG,WORKSPACE_CONFIG):validate_config(cfg)
        with self.assertRaises(ValueError):validate_config({**WORKSPACE_CONFIG,'max_updates_including_gate':121})

    def test_dataset_train_service_eval_identity_strictly_shared(self):
        fields=metadata(WSPACE:=WORKSPACE_PROFILE)
        dataset={'runs':[{},metadata(PROFILE),fields],'protocol':codec.ACTOR_VERSION}
        release={**fields,'protocol':codec.ACTOR_VERSION,'storage':storage.WORKSPACE_SPEC}
        self.assertEqual(require_pipeline_profile(release,dataset),WSPACE)
        require_same_pipeline(release,release)
        for key,value in (('protocol',protocol.VERSION),('action_codec',None),('execution_profile',PRECLOSE_PROFILE),('storage',storage.DIVERSE_SPEC)):
            bad={**release,key:value}
            with self.assertRaises(ValueError):require_pipeline_profile(bad,dataset)
            with self.assertRaises(ValueError):require_same_pipeline(release,bad)
        with self.assertRaises(ValueError):authorization_profile({'action_codec':codec.VERSION})
        tree=ast.parse((ROOT/'scripts/vlm_sft/native_eval_run.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='workspace_budget_ok']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(a) for a in calls[0].args],['token','servo.total_ticks','420 - controls','12 - index'])

    def test_actual_supervised_encoding_and_constrained_45_decoder(self):
        import torch
        from types import SimpleNamespace as NS
        from modeling import encode,decode
        _,_,_,old,raw=actor_fixtures.PoseProtocolTests().fixture()
        actor=protocol.actor_input(old['task'],old['active_instruction'],old['proprio'],old['current_rgb_sha256'],[],protocol=codec.ACTOR_VERSION)
        row,images=protocol.parse_request(protocol.request_payload(actor,raw,'base'),registered_instructions=[actor['active_instruction']])
        vocabulary=codec.tokens(codec.VERSION);ids={t:i+10 for i,t in enumerate(vocabulary)};eos=2
        tokenizer=NS(eos_token_id=eos,unk_token_id=-1,pad_token_id=0,
            convert_tokens_to_ids=lambda _:eos,convert_ids_to_tokens=lambda _:'<|im_end|>',
            encode=lambda token,**kw:[ids[token]],decode=lambda values,**kw:next(t for t,v in ids.items() if v==int(values[0])))
        processor=NS(tokenizer=tokenizer,apply_chat_template=lambda *a,**k:{'input_ids':torch.tensor([[1,3,4]]),'attention_mask':torch.ones((1,3),dtype=torch.long)})
        plain=encode(processor,row,images,supervised=False)
        trained=encode(processor,{**row,'target':codec.BODY_TOKENS[1]},images,supervised=True)
        self.assertTrue(torch.equal(plain['input_ids'],trained['input_ids'][:,:3]))
        self.assertEqual(trained['labels'].tolist(),[[-100,-100,-100,ids[codec.BODY_TOKENS[1]],eos]])
        with self.assertRaises(ValueError):encode(processor,{**protocol.training_row(old,'RIGHT_UP'),'target':codec.BODY_TOKENS[1]},images,supervised=True)
        seen=[]
        def generate(**kwargs):
            allowed=kwargs['prefix_allowed_tokens_fn'];prefix=kwargs['input_ids'][0]
            options=allowed(0,prefix);seen.append(options)
            selected=ids[codec.BODY_TOKENS[1]] if ids[codec.BODY_TOKENS[1]] in options else ids['RIGHT_UP']
            self.assertEqual(allowed(0,torch.cat([prefix,torch.tensor([selected])])),[eos])
            return torch.cat([prefix,torch.tensor([selected,eos])])[None,:]
        with patch.object(torch.Tensor,'to',lambda self,*a,**kw:self),patch('torch.cuda.synchronize'):
            self.assertEqual(decode(NS(generate=generate),processor,plain,action_codec=codec.VERSION)['prediction'],codec.BODY_TOKENS[1])
            self.assertEqual(decode(NS(generate=generate),processor,plain)['prediction'],'RIGHT_UP')
        self.assertEqual([len(v) for v in seen],[45,41])

    def test_all_six_actual_authorizations_and_checkpoint_protocol(self):
        from native_eval_run import require_evaluation,BUDGET,ROOT as EXPERIMENT
        from native_eval_prepare import SCHEMA
        from native_serve import validate_checkpoint
        from native_train import BASE
        from common import write_json,sha
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);gates=[]
            for task in (0,3):
                path=root/f'gate{task}.json';write_json(path,{'task':task,'gate_ok':True,'robot_geometry_guards':True,'gripper_completion_v1':True,'implementation_digest':'executor'});gates.append(str(path))
            for instance,episode in ((1,200),(71,247)):
                for variant in VARIANTS:
                    auth={**metadata(WORKSPACE_PROFILE),'schema':SCHEMA,'authorize_evaluation':True,'code_commit':'code','executor_digest':'executor',
                        'protocol':codec.ACTOR_VERSION,'dataset_sha256':'data','source':[1,episode,instance],'variant':variant,'physical_gpu':3,
                        'specified_hand':None,'oracle_actor_feedback':False,'reviewer':'parent','experiment_root':str(EXPERIMENT),'budget':BUDGET,
                        'engineering_gate_paths':gates,'storage':storage.WORKSPACE_SPEC}
                    output=EXPERIMENT/f'eval_t1_i{instance}_{variant}_v1';prepared={'source':auth['source']}
                    require_evaluation(auth,'code','executor',prepared,variant,output,'data')
                    for changes in ({'protocol':protocol.VERSION},{'action_codec':None},{'execution_profile':PRECLOSE_PROFILE},{'authorize_evaluation':False}):
                        with self.assertRaises(ValueError):require_evaluation({**auth,**changes},'code','executor',prepared,variant,output,'data')
            folder=root/'adapter_0120';folder.mkdir();(folder/'adapter_model.safetensors').write_bytes(b'fixture_not_model');write_json(folder/'adapter_config.json',{})
            identity={**metadata(WORKSPACE_PROFILE),'dataset_sha256':'data','protocol':codec.ACTOR_VERSION,'old_adapter_loaded':False,'base_model':BASE,'storage':storage.WORKSPACE_SPEC}
            write_json(root/'identity.json',identity)
            write_json(root/'result.json',{'status':'COMPLETE','optimizer_updates':120,'protocol':codec.ACTOR_VERSION,'dataset_sha256':'data','identity_sha256':sha(root/'identity.json'),
                'checkpoints':[{'step':120,'adapter_sha256':sha(folder/'adapter_model.safetensors'),'adapter_config_sha256':sha(folder/'adapter_config.json')}]})
            write_json(root/'restore_gate.json',{'passed':True})
            self.assertEqual(validate_checkpoint(root,'data',protocol=codec.ACTOR_VERSION)[0],folder)
            with self.assertRaises(ValueError):validate_checkpoint(root,'data')

    def test_actual_service_loader_rejects_changed_codec_on_each_response(self):
        import native_eval_run as runner
        from common import sha
        identity={**metadata(WORKSPACE_PROFILE),'protocol':codec.ACTOR_VERSION,'code_commit':'code','dataset_sha256':'data','physical_gpu':3,
            'port':8919,'max_calls':48,'storage':storage.WORKSPACE_SPEC,'adapter_sha256':'adapter','training_result_sha256':'training'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'service_v1').mkdir();path=root/'service_v1/identity.json';path.write_text(json.dumps(identity))
            auth={**metadata(WORKSPACE_PROFILE),'protocol':codec.ACTOR_VERSION,'storage':storage.WORKSPACE_SPEC,'service_identity_sha256':sha(path)}
            answer={**metadata(WORKSPACE_PROFILE),**{k:identity[k] for k in ('protocol','code_commit','dataset_sha256','adapter_sha256','training_result_sha256')},'storage_profile':storage.WORKSPACE_STORAGE_PROFILE}
            for changes in ({},{'action_codec':None},{'execution_profile':PRECLOSE_PROFILE},{'storage_profile':storage.DIVERSE_STORAGE_PROFILE}):
                streams=[io.BytesIO(json.dumps({**identity,'calls':0}).encode()),io.BytesIO(json.dumps({**answer,**changes}).encode())]
                with patch.object(runner,'ROOT',root),patch.object(runner,'urlopen',side_effect=streams):
                    _,remote=runner.load_service(auth,'code','data')
                    if changes:
                        with self.assertRaises(ValueError):remote({})
                    else:self.assertEqual(remote({}),answer)


if __name__=='__main__':unittest.main()

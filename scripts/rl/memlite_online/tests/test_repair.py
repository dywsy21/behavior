"""Run with python -m unittest discover -s scripts/rl/memlite_online/tests -v."""
import ast
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from recovery_recorder import RecoveryDetector,RecoveryRecorder
from training_config import DEFAULTS,load_training_config
from validate_recovery import validate_archive
from direct_ppo_core import sample_stochastic_flow,recompute_flow_step_log_probs
from direct_a4_flow import A4DirectPPO,ValueHead


def extracted_function(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


def context(verb='GRASP'):
    return dict(context_id='test-context-'+verb,parent_goal='test goal',
                active_skills_semantic_json=json.dumps([dict(verb=verb,target='obj')]),active_skills_text=verb)


def audit(q=0.,skill=0.,grasp='FALSE',success=False):
    return dict(official_q=q,skill_potential=skill,official_success=success,
                grasp_states={'obj':{'left':grasp,'right':'FALSE'}},
                literals=[dict(entities=['obj'],done=bool(q))])


class CorrectnessTests(unittest.TestCase):
    def test_all_short_terminal_batches_are_updated(self):
        for n in range(1,16):
            with self.subTest(n=n):
                calls=[];engine=types.SimpleNamespace(trainer=types.SimpleNamespace(
                    experiences={i:{} for i in range(n)},update=lambda *a:(calls.append(a) or {'updated':True})))
                fn=extracted_function(ROOT/'tools/serve_stage1_rl.py','dispatch',{'engine':engine})
                request=dict(kind='update',experience_ids=list(range(n)),advantages=[1.]*n,returns=[2.222]*n)
                self.assertEqual(fn(request),{'updated':True});self.assertEqual(len(calls),1)
                self.assertEqual(calls[0][1],list(range(n)))

    def test_bootstrap_then_infer_share_one_replan(self):
        ns={};fn=extracted_function(ROOT/'code/stage1_engine.py','ensure_context',ns)
        for chunk in (0,8,32,64):
            with self.subTest(chunk=chunk):
                calls=[];slot=dict(chunks=chunk,projection={'skill':'old'},planned_chunk=chunk-8)
                engine=types.SimpleNamespace(task='test',slots=[slot])
                def plan(obs,index):
                    calls.append(index);slot.update(projection={'skill':'new'},planned_chunk=slot['chunks'])
                engine.plan=plan
                before=fn(engine,{},0);after=fn(engine,{},0)
                self.assertIs(before,after);self.assertEqual(before['skill'],'new');self.assertEqual(calls,[0])

    def test_nonboundary_does_not_replan(self):
        fn=extracted_function(ROOT/'code/stage1_engine.py','ensure_context',{})
        engine=types.SimpleNamespace(task='t',slots=[dict(chunks=7,projection={'skill':'same'},planned_chunk=0)],
                                    plan=lambda *a:(_ for _ in ()).throw(AssertionError('unexpected')))
        self.assertEqual(fn(engine,{},0),{'skill':'same'})

    def test_stochastic_likelihood_identity_and_padding(self):
        torch.manual_seed(1);mask=torch.tensor([True,False,True])
        initial=torch.randn(2,4,3);initial[...,~mask]=0
        velocity=lambda x,t:0.1*x
        chain=sample_stochastic_flow(velocity,initial,0.02,mask,flow_steps=3)
        recomputed=recompute_flow_step_log_probs(velocity,chain.states,0.02,mask)
        torch.testing.assert_close(chain.step_log_prob,recomputed,rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(chain.states[...,~mask]),0)


class DetectorTests(unittest.TestCase):
    def test_alternating_arms_cannot_manufacture_stable_grasp(self):
        d=RecoveryDetector(capture_normal=True);events=[]
        for step in range(1,20):
            a=audit();a['grasp_states']['obj']={'left':'TRUE' if step%2 else 'FALSE',
                                               'right':'FALSE' if step%2 else 'TRUE'}
            events+=d.step(step,context(),a)
        self.assertFalse(d.stable);self.assertEqual(events,[])

    def test_tool_absent_from_final_goals_still_collected(self):
        d=RecoveryDetector(capture_normal=True);events=[]
        d.entity_bindings={'asset_broom':'obj'}
        ctx=context();ctx['active_skills_semantic_json']='[{"verb":"GRASP","target":"asset_broom"}]'
        for step in range(1,7):
            a=audit(grasp='TRUE');a['literals']=[dict(entities=['floor'],done=False)]
            events+=d.step(step,ctx,a)
        self.assertEqual(events[0]['kind'],'stable_grasp_observed')
        self.assertTrue(events[0]['active_target_matches'])
        self.assertFalse(events[0]['action_quality_verified'])

    def test_placing_one_object_does_not_hide_loss_of_another(self):
        d=RecoveryDetector();events=[]
        for step in range(1,7):d.step(step,context(),audit(grasp='TRUE'))
        ctx=context();ctx['active_skills_semantic_json']='[{"verb":"PLACE_IN","target":"other"}]'
        for step in range(7,13):events+=d.step(step,ctx,audit())
        self.assertEqual([e['kind'] for e in events],['grasp_loss_candidate'])

    def test_stable_loss_and_same_object_regrasp(self):
        d=RecoveryDetector();events=[]
        for step in range(1,7):events+=d.step(step,context(),audit(grasp='TRUE'))
        for step in range(7,13):events+=d.step(step,context(),audit())
        for step in range(13,19):events+=d.step(step,context(),audit(grasp='TRUE'))
        self.assertEqual([e['kind'] for e in events],['grasp_loss_candidate','regrasp_after_loss'])
        self.assertFalse(events[0]['failure_truth'])

    def test_unknown_grasp_never_means_release_or_success(self):
        d=RecoveryDetector();events=[]
        for step in range(1,7):events+=d.step(step,context(),audit(grasp='TRUE'))
        for step in range(7,20):events+=d.step(step,context(),audit(grasp='UNKNOWN'))
        self.assertEqual(events,[])

    def test_intended_placement_not_failure(self):
        d=RecoveryDetector();events=[]
        for step in range(1,7):events+=d.step(step,context(),audit(grasp='TRUE'))
        for step in range(7,13):events+=d.step(step,context('PLACE_IN'),audit())
        self.assertEqual(events,[])

    def test_handover_still_held(self):
        d=RecoveryDetector();events=[]
        for step in range(1,20):
            a=audit(grasp='TRUE')
            if step>6:a['grasp_states']['obj']={'left':'FALSE','right':'TRUE'}
            events+=d.step(step,context(),a)
        self.assertEqual(events,[])

    def test_goal_regression_debounced_then_restored(self):
        d=RecoveryDetector();events=[]
        for step,q in enumerate([.5,.5,0.,0.,0.,.5],1):events+=d.step(step,context(),audit(q=q))
        self.assertEqual([e['kind'] for e in events],['goal_regression_candidate','goal_progress_restored'])
        self.assertFalse(events[-1]['task_success'])

    def test_stall_uncertain_and_no_repeat_without_progress(self):
        d=RecoveryDetector(16);events=[]
        for step in range(1,60):events+=d.step(step,context(),audit())
        self.assertEqual(len(events),1);self.assertEqual(events[0]['kind'],'stalled_uncertain')
        self.assertFalse(events[0]['failure_truth'])

    def test_noncontiguous_clock_rejected(self):
        with self.assertRaises(ValueError):RecoveryDetector().step(2,context(),audit())


class RecorderTests(unittest.TestCase):
    def test_live_bindings_are_label_only_and_reset_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=self.make(tmp);rec.bind_scope(0,{'obj':'asset_1','future':None})
            self.roll(rec)
            for path in Path(tmp).glob('*.zip'):
                with zipfile.ZipFile(path) as archive:
                    m=json.loads(archive.read('manifest.json'))
                    self.assertEqual(m['label_only_entity_bindings']['by_asset_name'],{'asset_1':'obj'})
                    self.assertNotIn('label_only_entity_bindings',m['episode'])
                    self.assertNotIn('entity_bindings',m['rgb_anchors'])
            with self.assertRaises(ValueError):rec.bind_scope(0,{'obj':'other'})
    def make(self,path,**overrides):
        settings=DEFAULTS|dict(recovery_pre_controls=8,recovery_post_controls=8,
            recovery_max_controls=32,recovery_stall_controls=8,disk_reserve_gib=.0001)|overrides
        rec=RecoveryRecorder(path,1,settings,dict(run='test',source_commit='a'*40))
        metadata=dict(episode_id='test-env0',task='t',instance_id=1,split='train',policy_seed=17,
                      source_commit='a'*40,resume_checkpoint_sha256='b'*64)
        rec.begin([metadata]);return rec

    def roll(self,rec,steps=32,terminal=True):
        for step in range(steps):
            if step%16==0:
                obs=dict(images={k:np.full((3,16,20),step,dtype=np.uint8) for k in ('head_rgb','left_wrist_rgb','right_wrist_rgb')},
                         state={'left_arm':np.zeros(7)})
                rec.on_chunk(0,obs,context(),step//16,23)
            rec.before_action(0,{'robot::proprio':np.full(61,step,dtype=np.float32)},np.ones(23)*.1)
            rec.confirm_applied(0,np.ones(23)*.1)
            rec.observe(0,{'robot::proprio':np.full(61,step+1,dtype=np.float32)},0.,audit(),False,terminal and step==steps-1)

    def test_zip_content_alignment_and_pending_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=self.make(tmp);self.roll(rec);rec.close()
            archives=list(Path(tmp).glob('*.zip'));self.assertGreaterEqual(len(archives),1)
            for f in archives:
                receipt=validate_archive(f);self.assertFalse(receipt['bc_eligible'])
                self.assertEqual(receipt['human_review'],'pending')

    def test_nontrain_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=self.make(tmp);row=deepcopy(rec.slots[0]['metadata']);row['split']='public_test'
            with self.assertRaises(ValueError):rec.begin([row])

    def test_quota_never_publishes_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=self.make(tmp,recorder_quota_gib=1e-9);self.roll(rec);rec.close()
            self.assertFalse(list(Path(tmp).glob('*.zip')))
            self.assertGreater(rec.counts['quota_rejected'],0)

    def test_no_false_after_state_on_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=self.make(tmp);self.roll(rec,12,False)
            rec.before_action(0,{'robot::proprio':np.full(61,12,dtype=np.float32)},np.zeros(23))
            rec.close('interrupted')
            f=next(Path(tmp).glob('*.zip'));receipt=validate_archive(f)
            with zipfile.ZipFile(f) as z:m=json.loads(z.read('manifest.json'))
            self.assertEqual(m['end_control_step'],12);self.assertEqual(m['stop_reason'],'interrupted')

    def test_mismatched_applied_command_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=self.make(tmp);self.roll(rec,1,False)
            rec.before_action(0,{'robot::proprio':np.ones(61)},np.zeros(23))
            with self.assertRaises(ValueError):rec.confirm_applied(0,np.ones(23))
            with self.assertRaises(ValueError):rec.observe(0,{'robot::proprio':np.ones(61)},0,audit(),False,False)


class PPOUpdateTests(unittest.TestCase):
    def run_update(self,n,adv,scale=10,force_reject=False):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self))
            stack.enter_context(patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self))
            for name in ('synchronize','memory_allocated','max_memory_allocated'):
                stack.enter_context(patch.object(torch.cuda,name,lambda *a,**k:0))
            model=torch.nn.Module();model.action_expert=torch.nn.Linear(1,1,bias=False)
            with torch.no_grad():model.action_expert.weight.fill_(.2)
            policy=torch.nn.Module();policy.model=model
            trainer=A4DirectPPO(policy,torch.tensor([False]),Path(tmp),actor_lr=1e-7,target_kl=.02)
            trainer._ensure_critic(torch.ones(1,1));reference=model.action_expert.weight.detach().clone()
            def evaluate(self,batch,chains,old,advantages,returns):
                delta=(model.action_expert.weight-reference).sum()*scale
                values=self.critic(torch.ones(len(advantages),1))
                policy_loss=-delta*advantages.mean();vl=.5*(values-returns).square().mean()
                ratio=float(delta.detach().exp());kl=ratio-1-float(delta.detach())
                return policy_loss+.5*vl,dict(policy_loss=float(policy_loss.detach()),value_loss=float(vl.detach()),
                    ratio_mean=ratio,approx_kl=max(0.,kl),clip_fraction=1.0 if force_reject else float(abs(ratio-1)>.2))
            trainer.evaluate_prepared=types.MethodType(evaluate,trainer)
            for i in range(n):trainer.experiences[i]=dict(observation={},local_subgoal={},chain=torch.zeros(2,1,1),
                old_step_log_prob=torch.zeros(1),old_value=0.)
            infer=types.SimpleNamespace(prepare_training_batch=lambda *a:{},_branch_context=nullcontext)
            if force_reject:
                with self.assertRaisesRegex(RuntimeError,'no safe direct PPO step'):
                    trainer.update(infer,list(range(n)),[adv]*n,[.8]*n)
                torch.testing.assert_close(model.action_expert.weight,reference,rtol=0,atol=0)
                self.assertEqual(len(trainer.actor_optimizer.state),0)
                self.assertEqual(trainer.update_count,0)
                self.assertEqual(len(trainer.experiences),n)
                return
            result=trainer.update(infer,list(range(n)),[adv]*n,[.8]*n)
            self.assertEqual(trainer.experiences,{})
            self.assertIn('raw_advantages',result);self.assertIn('value_loss_before',result)
            self.assertEqual(result['update'],1)
            return result

    def test_real_autograd_short_batches(self):
        for n in (1,2,8,15,64):
            with self.subTest(n=n):
                r=self.run_update(n,1.);self.assertTrue(r['actor_updated']);self.assertEqual(r['actor_updates'],1)

    def test_zero_advantage_updates_critic_without_fake_actor_step(self):
        r=self.run_update(1,0.);self.assertFalse(r['actor_updated']);self.assertEqual(r['actor_updates'],0)
        self.assertEqual(r['accepted_actor_lr'],0.)

    def test_unsafe_step_backtracks_actual_adam_state(self):
        r=self.run_update(2,1.,scale=1e7)
        self.assertGreater(r['optimizer_steps'],1)
        self.assertLess(r['accepted_actor_lr'],1e-7)
        self.assertLessEqual(r['post_update']['mean_approx_kl'],.02)

    def test_rejection_restores_weights_and_optimizer(self):
        self.run_update(2,1.,force_reject=True)


if __name__=='__main__':unittest.main()

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from causal_skill_probe import CausalSkillProbe,validate_probe_histories,load_causal_probe_metadata
from recovery_causal_handover import teacher_prefix_projection
from recovery_corpus import digest
from g05.utils.memlite_skill_protocol import append_b_memory_idempotent
from test_recovery_causal_handover import fixture


class Planner:
    def ensure_context(self,session,identity,observation,*,validate_low_goal,interval_controls):
        if not session.planning_due(identity,interval_controls=interval_controls):
            goal=session.low_goal(identity);validate_low_goal(goal)
            return dict(goal=goal,reused=True,event=None,causal_input=None,control_step=session.feedback.control_step,
                revision=session.revision,physical_success_asserted=False,observer_feedback_mode='shadow_unknown_v1')
        token,causal=session.begin_planning(identity,session.feedback.control_step)
        event=deepcopy(self.event)
        event['memory_update']=append_b_memory_idempotent(session.memory,session.previous_intent,task_name=identity.task)
        validate_low_goal(session.stage(identity,token,event))
        return dict(goal=session.commit(identity,token),reused=False,causal_input=causal,event=event,
            control_step=session.feedback.control_step,revision=session.revision,
            physical_success_asserted=False,observer_feedback_mode='shadow_unknown_v1')


class ProbeTests(unittest.TestCase):
    def test_registered_joint_probe_keeps_real_histories_low_and_fixed_scores(self):
        from recovery_corpus import file_sha
        repo=Path(__file__).resolve().parents[4]
        cfg=json.loads((repo/'configs/recovery_sft/a800_causal_joint_shadow_probes_v1.json').read_text())
        fixed=json.loads((repo/'configs/recovery_sft/a800_later_action_replication_later_actions_v1.json').read_text())
        spec=cfg['causal_planner'];models=json.loads((repo/spec['models_config']).read_text())
        self.assertEqual(file_sha(repo/spec['models_config']),spec['models_config_sha256'])
        self.assertEqual(models['low'],cfg['model']);self.assertEqual(cfg['model'],fixed['model'])
        self.assertEqual(cfg['evaluation_seeds'],fixed['evaluation_seeds'][:2])
        self.assertEqual(len(cfg['cases']),2)
        for key,case in cfg['cases'].items():
            self.assertEqual(case,fixed['cases'][key]);self.assertEqual(case['sim']['kind'],'recovery')
        self.assertEqual(cfg['learning_rounds'],0)
        self.assertFalse(spec['observer_predictions_enabled'])
        self.assertEqual(spec['scoring'],'original_restored_skill_endpoint_not_RL_reward')

    def test_two_case_exception_is_only_explicit_readonly_not_training(self):
        from skill_rounds import SkillRounds,SkillEvaluationRounds
        cases={'a':{},'b':{}}
        with self.assertRaises(ValueError):SkillRounds(cases,[1,2],rounds=1,run='no')
        with self.assertRaises(ValueError):SkillEvaluationRounds(cases,[1,2],run='no')
        rounds=SkillEvaluationRounds(cases,[1,2],run='joint',allow_two_logged_cases=True)
        self.assertEqual(len(rounds.jobs),4)
        for case in cases:
            for _ in range(2):
                job=rounds.take(case);rounds.complete(job['id'],[])
        with self.assertRaises(ValueError):rounds.advance(optimizer_completed=True)
        rounds.advance();self.assertEqual(rounds.phase,'finished')

    def test_joint_recipe_cannot_admit_training_or_fake_observer_truth(self):
        self.assertIsNone(load_causal_probe_metadata({},Path('.')))
        base=dict(schema='short_skill_sft_evaluation_a800_v1',user_goal_authorized=True,
            learning_rounds=0,root='/unused',simulator_reset_protocol='fresh_process_each_episode_v1')
        spec=dict(protocol='recorded_handover_shadow_v1',models_config='unread',models_config_sha256='a'*64,
            inputs_manifest='unread',inputs_manifest_sha256='b'*64,planning_interval_controls=128,
            observer_predictions_enabled=False,scoring='original_restored_skill_endpoint_not_RL_reward')
        for bad in (None,{},dict(spec,observer_predictions_enabled=True),
                    dict(spec,planning_interval_controls=16),dict(spec,scoring='train_reward')):
            with self.assertRaises(ValueError):load_causal_probe_metadata(dict(base,causal_planner=bad),Path('.'))
        with self.assertRaises(ValueError):
            load_causal_probe_metadata(dict(base,schema='short_skill_rl_a800_v1',learning_rounds=1,causal_planner=spec),Path('.'))

    def setUp(self):
        identity,plans,controls=fixture()
        projection=teacher_prefix_projection(plans,controls,at_control_step=32)
        self.models=identity.models;self.cases={};rows=[];self.physical=[];self.jobs=[]
        self.log=[];self.low_goals=[]
        self.observation=dict(images={k:np.zeros((3,16,16),np.uint8) for k in
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')},proprio=np.zeros(61,np.float32))
        planner=Planner();planner.event=projection['events'][0]['event']
        for instance in (3,4):
            case_key='some_task_'+str(instance)
            case=dict(task='some_task',instance_id=instance,original_split='train',recovery_split='train',
                start_control=32,end_control=240,manifest_sha256='e'*64,
                sim=dict(kind='recovery',proof_sha256='f'*64),semantic_bundle='not_an_actor_target')
            self.cases[case_key]=case
            rows.append(dict(case=case_key,task='some task',instance=instance,record=dict(control_step=32),
                issued_prefix=deepcopy(projection),issued_prefix_sha256=digest(projection),
                source_branch_manifest_sha256='e'*64,source_replay_proof_sha256='f'*64))
            job=dict(case=case_key,phase='evaluation',round=0,id='job-'+str(instance),seed=17+instance)
            physical=SimpleNamespace(case=case,identity=SimpleNamespace(session='sharedservice',episode=job['id']),
                version=0,sha=self.models.low,rollout=SimpleNamespace(step=32,ended=False))
            self.jobs.append(job);self.physical.append(physical)
        self.manifest=dict(schema='actual_local_causal_handover_inputs_v1',rows=rows,
            role='TRAIN_engineering_only_never_SFT_calibration_or_test')
        low=SimpleNamespace(prepare=lambda obs,goal:self.low_goals.append(deepcopy(goal)))
        self.probe=CausalSkillProbe(self.cases,self.manifest,self.models,planner,low,
            log_event=lambda kind,value:self.log.append((kind,deepcopy(value))))

    def start(self,i=0):
        p,j=self.physical[i],self.jobs[i]
        self.probe.begin(p,j);goal=self.probe.goal(p,j,self.observation)
        return p,j,goal

    def apply(self,p,j,n=16):
        actions=np.arange(16*23,dtype=np.float32).reshape(16,23)/1000
        self.probe.offer(p,j,actions,16)
        ack=[dict(control_step=p.rollout.step+k+1,simulator_apply_ack=True,
            action_executed_raw23=actions[k].tolist(),physical_evidence={'truth':'must_not_leak'},
            official_terminal=False,official_truncated=False) for k in range(n)]
        p.rollout.step+=n
        self.probe.acknowledge(p,j,ack)
        return ack

    def test_actual_ack_conversion_partial_tail_and_no_truth_input(self):
        p,j,goal=self.start();self.assertNotEqual(goal['semantic_bundle'],p.case['semantic_bundle'])
        ack=self.apply(p,j,3)
        session,_=self.probe.active[j['id']]
        self.assertEqual(session.feedback.control_step,35)
        self.assertEqual(self.probe.applied_controls,3)
        for _,entry in self.log:
            self.assertNotIn('must_not_leak',str(entry))
        with self.assertRaises(ValueError):self.probe.acknowledge(p,j,ack)
        with self.assertRaises(ValueError):self.probe.close(p,j)
        p.rollout.ended=True;self.probe.close(p,j)
        self.assertEqual(self.probe.audit()['active_episodes'],0)
        self.assertEqual(self.probe.audit()['observer_predictions'],0)
        with self.assertRaises(ValueError):self.probe.begin(p,j)

    def test_task_episode_isolation_and_real128_control_cadence(self):
        p,j,goal=self.start();other,other_job,_=self.start(1)
        for _ in range(8):
            self.assertEqual(self.probe.goal(p,j,self.observation)['semantic_bundle'],goal['semantic_bundle'])
            self.apply(p,j)
        self.assertEqual(self.probe.generations,2)
        self.assertEqual(self.probe.active[other_job['id']][0].feedback.control_step,32)
        self.probe.goal(p,j,self.observation)
        self.assertEqual(self.probe.generations,3)
        self.assertEqual(self.probe.active[j['id']][0].feedback.control_step,160)
        with self.assertRaises(ValueError):self.probe.goal(p,other_job,{})

    def test_unbound_history_current_boundary_and_placement_rejected(self):
        for mutate in (lambda f:f['rows'][0].update(instance=99),
                       lambda f:f['rows'][0].update(source_replay_proof_sha256='a'*64),
                       lambda f:f['rows'][0]['issued_prefix'].update(control_step=33),
                       lambda f:f['rows'].append(deepcopy(f['rows'][0]))):
            bad=deepcopy(self.manifest);mutate(bad)
            with self.assertRaises(ValueError):validate_probe_histories(bad,self.cases)
        cases=deepcopy(self.cases);cases[self.jobs[0]['case']]['sim']['kind']='placement'
        with self.assertRaises(ValueError):validate_probe_histories(self.manifest,cases)

    def test_train_or_mismatched_policy_forbidden(self):
        p,j=self.physical[0],self.jobs[0]
        for change in (dict(phase='train'),dict(round=1),dict(id='foreign')):
            with self.assertRaises(ValueError):self.probe.begin(p,dict(j,**change))
        p.sha='wrong'
        with self.assertRaises(ValueError):self.probe.begin(p,j)

    def test_bad_ack_or_inflight_planning_does_not_advance(self):
        p,j,_=self.start();actions=np.zeros((16,23),dtype=np.float32)
        self.probe.offer(p,j,actions,16)
        with self.assertRaises(ValueError):self.probe.goal(p,j,{})
        ack=[dict(control_step=34,simulator_apply_ack=True,action_executed_raw23=[0.]*23)]
        p.rollout.step=34
        with self.assertRaises(ValueError):self.probe.acknowledge(p,j,ack)
        self.assertEqual(self.probe.active[j['id']][0].feedback.control_step,32)


if __name__=='__main__':unittest.main()

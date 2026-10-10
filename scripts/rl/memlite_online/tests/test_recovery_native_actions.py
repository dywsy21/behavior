from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_native_actions import (checked_action_window,native_low_raw,
    same_event_supplement_rows,SameEventLearnerSupplement,validate_native_training_supplement)
from recovery_corpus import digest


class NativeLearnerActionTests(unittest.TestCase):
    def rows(self):
        return [dict(control_step=t,simulator_apply_ack=True,policy_version=25,action_source='train',
            action_executed_raw23=np.full(23,t/100,dtype=np.float32).tolist(),
            skill_reward=dict(terminated=t==132,truncated=False,identity={'episode':'bound','bundle':'same'}))
            for t in range(101,133)]

    def read(self, rows, at=100):
        return checked_action_window(rows,at=at,identity={'episode':'bound','bundle':'same'},policy_version=25)

    def test_native_post_control_clock_exact23_no_terminal_padding(self):
        rows=self.rows();action,sha=self.read(rows)
        self.assertEqual(action.shape,(32,23));self.assertEqual(action.dtype,np.float32)
        self.assertEqual(action.tolist(),[r['action_executed_raw23'] for r in rows])
        self.assertEqual(sha,digest(action.tolist()))
        with self.assertRaises(ValueError):self.read(rows,101)
        with self.assertRaises(ValueError):self.read(rows[:-1])

    def test_cross_source_policy_and_nontraining_actions_fail(self):
        for mutate in (lambda r:r[10].update(simulator_apply_ack=False),
                       lambda r:r[10].update(policy_version=26),
                       lambda r:r[10].update(action_source='evaluation'),
                       lambda r:r[10]['skill_reward']['identity'].update(episode='foreign'),
                       lambda r:r[10]['skill_reward']['identity'].update(bundle='different'),
                       lambda r:r[10]['skill_reward'].update(terminated=True)):
            rows=self.rows();mutate(rows)
            with self.assertRaises(ValueError):self.read(rows)

    def test_dense_or_padded_or_nonfinite_or_requantized_actions_fail(self):
        for change in ([0.]*27,[0.]*22,[float('nan')]*23,[.1]*23):
            rows=self.rows();rows[0]['action_executed_raw23']=change
            with self.assertRaises(ValueError):self.read(rows)

    def test_duplicate_or_float_clock_fails(self):
        rows=self.rows()
        with self.assertRaises(ValueError):self.read(rows+[deepcopy(rows[-1])])
        rows[0]['control_step']=101.
        with self.assertRaises(ValueError):self.read(rows)

    def test_raw_low_projection_only_current_observation_and_same_skill(self):
        import importlib.util
        import types
        import torch
        from unittest.mock import patch
        from recovery_corpus import canonical
        # Execute the real dependency-free label module without the unrelated
        # eager g05.data dataset registry (local CPU unit env has no OmegaConf).
        name='g05.data.memlite_stage1_labels'
        spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[4]/'src/g05/data/memlite_stage1_labels.py')
        labels=importlib.util.module_from_spec(spec);spec.loader.exec_module(labels)
        stub=patch.dict(sys.modules,{'g05.data':types.ModuleType('g05.data'),name:labels})
        stub.start();self.addCleanup(stub.stop)
        observation=dict(proprio=np.zeros(61,dtype=np.float32),images={k:np.zeros((3,32,32),dtype=np.uint8)
            for k in ('head_rgb','left_wrist_rgb','right_wrist_rgb')})
        goal=dict(task='some task',parent_goal='Task goal: some task',semantic_bundle=canonical([
            dict(verb='GRASP',target='cup',source='',destination='',target_part='',arm='LEFT',unbound_relation='')]))
        action=np.arange(32*23,dtype=np.float32).reshape(32,23)
        reader=types.SimpleNamespace(read=lambda _: (observation,action,goal))
        config=dict(raw_shape=dict(state=[dict(key='actual_state',start_index=0,raw_shape=61)],
            action=[dict(key='actual_controls',start_index=0,raw_shape=23)]))
        raw=native_low_raw(reader,0,config)
        self.assertEqual(raw['model_projection']['memlite_branch'],'low')
        self.assertTrue(torch.equal(raw['action']['actual_controls'],torch.from_numpy(action)))
        self.assertFalse(raw['action_is_pad'].any())
        for forbidden in ('reward','physical_evidence','source_policy_sha256','source_group','outcome_target'):
            self.assertNotIn(forbidden,raw)
        reader.read=lambda _: (dict(observation,reward=1),action,goal)
        with self.assertRaises(ValueError):native_low_raw(reader,0,config)


class SameEventSupplementTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(candidate=dict(sample_id=f'base{i}',task=f'task_{i}',source_group=f'task {i}:1',
            split='train',actor_input=dict(parent_goal='same parent',issued_skills_semantic_json='[{"verb":"OPEN_DOOR","target":"washer"}]')),
            approval=dict(pool='action',event_id=f'event{i}')) for i in range(19)]
        native=[dict(sample_id=f'{i+100:064x}',task='task_0',source_group='task 0:1',
            original_split='train',recovery_split='train',parent_goal='same parent',
            semantic_bundle='[{"target":"washer","verb":"OPEN_DOOR"}]',
            action_source='reviewed_learner_late_correction_not_expert') for i in range(5)]
        return rows,native

    def test_multiple_learner_windows_do_not_add_events_or_change_mixture(self):
        from recovery_sft_data import finite_mixture_schedule
        rows,native=self.fixture();extended=rows+same_event_supplement_rows(rows,native)
        experts={str(t):list(range(t*100,t*100+100)) for t in range(100)}
        kwargs=dict(batch_size=8,maximum_event_passes=20,seed=17,allow_extended_event_fit=True,
                    anchor_selection_protocol='independent_anchor_rng_v1')
        old=list(finite_mixture_schedule(rows,experts,**kwargs))
        new=list(finite_mixture_schedule(extended,experts,**kwargs))
        self.assertEqual(len(old),200);self.assertEqual(len(new),len(old))
        native_draws=0
        for left,right in zip(old,new):
            self.assertEqual(left['new_events'],right['new_events'])
            self.assertEqual(left['expert_count'],right['expert_count'])
            self.assertEqual(left['new_count'],right['new_count'])
            for (lk,li),(rk,ri) in zip(left['rows'],right['rows']):
                self.assertEqual(lk,rk)
                if lk=='expert':self.assertEqual(li,ri)
                else:
                    self.assertEqual(rows[li]['approval']['event_id'],extended[ri]['approval']['event_id'])
                    native_draws+=ri>=len(rows)
        self.assertGreater(native_draws,0);self.assertLessEqual(native_draws,20)

    def test_split_source_task_parent_skill_and_permission_are_all_bound(self):
        for key,value in [('original_split','eval'),('recovery_split','dev'),
                ('source_group','other:1'),('task','foreign'),('parent_goal','wrong parent'),
                ('semantic_bundle','[{"verb":"OPEN_DOOR","target":"other"}]'),
                ('action_source','expert')]:
            rows,native=self.fixture();native[0][key]=value
            with self.assertRaises(ValueError):same_event_supplement_rows(rows,native)

    def test_ambiguous_event_and_duplicate_and_empty_are_rejected(self):
        rows,native=self.fixture();other=deepcopy(rows[0]);other['candidate']['sample_id']='another'
        other['approval']['event_id']='other_event'
        for base,extra in [(rows+[other],native),(rows,native+[native[0]]),(rows,[])]:
            with self.assertRaises(ValueError):same_event_supplement_rows(base,extra)
        native[0]['sample_id']=rows[0]['candidate']['sample_id']
        with self.assertRaises(ValueError):same_event_supplement_rows(rows,native)

    def test_dev_cannot_be_wrapped_and_existing_samples_keep_dispatch(self):
        from types import SimpleNamespace
        rows,native=self.fixture()
        class Base:
            split='train';config={}
            def __len__(self):return len(rows)
            def __getitem__(self,i):return ('unaltered_original',i)
        base=Base();base.rows=rows
        wrapped=SameEventLearnerSupplement(base,SimpleNamespace(rows=native))
        self.assertEqual(len(wrapped),24)
        for i in range(19):self.assertEqual(wrapped[i],base[i])
        base.split='dev'
        with self.assertRaises(ValueError):SameEventLearnerSupplement(base,SimpleNamespace(rows=native))

    def test_no_implicit_supplement_ticket(self):
        self.assertIsNone(validate_native_training_supplement({'files':{}},{'L0':{}},[]))
        with self.assertRaises(ValueError):
            validate_native_training_supplement({'files':{'native_manifest':{}}},{'L0':{}},[])
        for protocol in ('replicate_events','same_existing_source_event_v1'):
            with self.assertRaises(ValueError):
                validate_native_training_supplement({'component':'H1','files':{}},
                    {'L0':{'native_learner_supplement':dict(protocol=protocol,manifest={},owner_review={})}},[])

    def test_native_launch_receipt_binds_actual_train_source_stats_admission_and_masks(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from recovery_corpus import file_sha
        rows,native=self.fixture()
        for i,r in enumerate(native):r['actions_sha256']=f'{i+200:064x}'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('manifest.json','owner.json'):(root/name).write_text('{}')
            spec=dict(protocol='same_existing_source_event_v1',
                manifest=dict(path='manifest.json',sha256=file_sha(root/'manifest.json')),
                owner_review=dict(path='owner.json',sha256=file_sha(root/'owner.json')))
            recipe=dict(root=directory,stats_sha256='s'*64,L0=dict(native_learner_supplement=spec))
            ticket=dict(component='L0',source_commit='frozen_source',files={
                'native_manifest':dict(path=str(root/'manifest.json'),sha256=spec['manifest']['sha256']),
                'native_owner_review':dict(path=str(root/'owner.json'),sha256=spec['owner_review']['sha256']),
                'recipe':dict(sha256='recipe_hash'),'admission':dict(sha256='admission_hash')})
            audit=dict(schema='native_learner_processor_audit_v1',status='passed',source_commit='frozen_source',
                optimizer_steps=0,oracle_inputs=False,stats_sha256=recipe['stats_sha256'],training_processor_checked=True,
                recipe_sha256='recipe_hash',admission_sha256='admission_hash',
                same_event_schedule_check=dict(unchanged_expert_event_rank_order=True),
                manifest_sha256=spec['manifest']['sha256'],owner_review_sha256=spec['owner_review']['sha256'],
                samples=[dict(sample_id=r['sample_id'],actions_sha256=r['actions_sha256'],shape=[32,27],
                    padding_indices=[7,8,17,18],train_masks_verified=True) for r in native])
            def call(value):
                path=root/'audit.json';path.write_text(json.dumps(value))
                ticket['files']['native_processor_audit']=dict(path=str(path),sha256=file_sha(path))
                with patch('recovery_native_actions.NativeLearnerActionReader',return_value=SimpleNamespace(rows=native)):
                    return validate_native_training_supplement(ticket,recipe,rows)
            self.assertEqual(call(audit).rows,native)
            for field,value in [('source_commit','stale'),('optimizer_steps',1),('oracle_inputs',True),
                    ('stats_sha256','foreign'),('training_processor_checked',False),('recipe_sha256','different'),
                    ('admission_sha256','different'),('same_event_schedule_check',{}),('samples',[])]:
                bad=deepcopy(audit);bad[field]=value
                with self.assertRaises(ValueError):call(bad)
            bad=deepcopy(audit);bad['samples'][0]['padding_indices']=[0,1,2,3]
            with self.assertRaises(ValueError):call(bad)


if __name__=='__main__':unittest.main()

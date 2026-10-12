import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
import recovery_periodic_data as module


def row():
    return dict(identity=dict(original_candidate_index=7,task=0,split='train',episode=2,frame=128,
                              raw_episode=20,instance=1),human_admission=False,
        teacher_kind='annotation_periodic_command_trace_not_actor_rollout',annotation_trace=dict(period=128),
        projection=dict(memlite_branch='high',schema_version=6,outcome_supervision_mask=False,
            low_action_supervision_mask=False,known_previous_outcome='UNKNOWN',outcome_target='UNKNOWN',
            task_complete=False,next_decision='EXECUTE',memory='past only'))


class PeriodicDataTests(unittest.TestCase):
    def test_no_result_low_or_test_labels_can_enter_periodic_pool(self):
        self.assertEqual(module.validate_periodic_row(row()), (7,0))
        for field,value in (('memlite_branch','low'),('outcome_supervision_mask',True),
                            ('low_action_supervision_mask',True),('outcome_target','SUCCEEDED'),
                            ('known_previous_outcome','FAILED'),('task_complete',True),('next_decision','STOP')):
            changed=row(); changed['projection'][field]=value
            with self.assertRaises(ValueError): module.validate_periodic_row(changed)
        changed=row(); changed['identity']['split']='eval'
        with self.assertRaises(ValueError): module.validate_periodic_row(changed)

    def fixture(self):
        binding=dict(component='H1',parent_sha256='parent',admission_sha256='data',world_size=8,micro_batch=1)
        recovery=[dict(candidate=dict(source_group=f'g{i}',split='train'),approval=dict(event_id=f'e{i}')) for i in range(2)]
        batch=dict(event_pass=0, rows=[['expert',i] for i in range(3)]+[['new',1]]+
            [['expert',i] for i in range(3,6)]+[['new',0]],new_events=[['g0','e0'],['g1','e1']],expert_count=6,new_count=2)
        control=dict(binding=binding,schedule=[batch,copy.deepcopy(batch)])
        pools={str(i):[100+i,200+i] for i in range(6)}
        return binding,recovery,control,pools

    def test_remap_preserves_every_recovery_slot_and_task_not_only_counts(self):
        binding,recovery,control,pools=self.fixture(); original=copy.deepcopy(control)
        def remap(): return module.remap_control_schedule(control,pools,lambda i:i,seed=17,recovery_rows=recovery,binding=binding)
        mapped=remap(); self.assertEqual(mapped,remap()); self.assertEqual(control,original)
        for old,new in zip(control['schedule'],mapped):
            self.assertEqual(old['new_events'],new['new_events'])
            for before,after in zip(old['rows'],new['rows']):
                if before[0]=='new': self.assertEqual(before,after)
                else: self.assertEqual(before[1],after[1]%100)
        self.assertEqual(len({i for batch in mapped for kind,i in batch['rows'] if kind=='expert'}),12)

    def test_control_mismatch_or_recovery_dev_or_missing_task_refused(self):
        binding,recovery,control,pools=self.fixture()
        cases=[(dict(binding,parent_sha256='foreign'),recovery,pools),
               (binding,[dict(recovery[0],candidate=dict(source_group='g0',split='dev')),recovery[1]],pools),
               (binding,recovery,{k:v for k,v in pools.items() if k!='3'})]
        for b,r,p in cases:
            with self.assertRaises(ValueError): module.remap_control_schedule(control,p,lambda i:i,seed=17,recovery_rows=r,binding=b)

    def test_actual_raw_overlay_rejects_unknown_index_and_never_mutates_source(self):
        source=row(); pool=SimpleNamespace(by_index={7:source},by_task={'0':[7]},release=SimpleNamespace(root=Path('/accepted')))
        class FakeSource:
            def __init__(self):
                self.release=Path('/accepted');self.branch='high';self.split='train'
                self.candidates={7:(0,8)};self.task_ids={7:0};self.config={'raw_shape':{'action':[{'key':'base','raw_shape':3}]}}
            def episode(self,_):return {}
            def raw(self,index,*,images=True):
                return {'model_projection':{'memory':'dense'}},dict(episode=2,task=0,frame=128,raw_episode=20,instance=1)
        wrapped=module.with_periodic_teacher(FakeSource,pool)()
        with patch.object(module,'validate_candidate_projection',return_value=source['projection']):
            first,_=wrapped.raw(7);first['model_projection']['memory']='corrupt caller'
            second,_=wrapped.raw(7)
        self.assertEqual(second['model_projection']['memory'],'past only')
        self.assertTrue(second['action_is_pad'].all());self.assertEqual(tuple(second['action']['base'].shape),(32,3))
        with self.assertRaises(ValueError): wrapped.raw(8)

    def test_champion_capacity_preserved_plus_emergency_checkpoint(self):
        result=module.checkpoint_reservation(26593048600,360,120,775<<30)
        self.assertEqual(result['planned_checkpoints'],4)
        self.assertGreater(result['required_free_bytes'],775<<30)
        self.assertGreater(result['runtime_stop_free_bytes'],775<<30)
        with self.assertRaises(ValueError): module.checkpoint_reservation(26593048600,360,120,256<<30)

    def test_foreign_artifact_path_refused(self):
        with self.assertRaises(ValueError): module.bound('/accepted','../elsewhere')


if __name__=='__main__': unittest.main()

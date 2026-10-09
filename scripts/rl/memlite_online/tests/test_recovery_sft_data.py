from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_sft_data import finite_mixture_schedule,require_training_pool
from recovery_corpus import file_sha


class DataTests(unittest.TestCase):
    def test_blocked_pool_rejected_before_torch_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'admission.json'
            p.write_text(json.dumps(dict(schema='recovery_admission_v1',pools={'action':dict(training_ready=False,blockers=['no verified actions'])})))
            with self.assertRaisesRegex(ValueError,'BLOCKED'):
                require_training_pool(tmp,'action',file_sha(p))
            with self.assertRaisesRegex(ValueError,'unpinned'):
                require_training_pool(tmp,'action','wrong')

    def test_no_frame_inflation_or_repeat_filled_tail(self):
        rows=[]
        for event,count in [('e0',100),('e1',2),('e2',1)]:
            rows += [dict(candidate=dict(split='train',source_group='t:1'),approval=dict(pool='action',event_id=event)) for _ in range(count)]
        args=dict(new_rows=rows,expert_by_task={'t':[0,1],'u':[2,3]},batch_size=64)
        schedule=list(finite_mixture_schedule(**args))
        self.assertEqual(schedule,list(finite_mixture_schedule(**args)))
        counts=Counter(tuple(event) for batch in schedule for event in batch['new_events'])
        self.assertEqual(set(counts.values()),{5})
        self.assertEqual(len(schedule),5)
        self.assertTrue(all(b['new_count']==3 and b['expert_count']==7 and len(b['rows'])==10 for b in schedule))
        for batch in schedule:
            experts=[idx for source,idx in batch['rows'] if source=='expert']
            self.assertTrue(any(x<2 for x in experts) and any(x>=2 for x in experts))

    def test_full_global_batch_and_protected_rejection(self):
        rows=[dict(candidate=dict(split='train',source_group=f't:{i}'),approval=dict(pool='action',event_id=str(i))) for i in range(25)]
        batches=list(finite_mixture_schedule(rows,{'t':[0],'u':[1]},maximum_event_passes=1))
        self.assertEqual(len(batches[0]['rows']),64)
        self.assertEqual(batches[0]['new_count'],19)
        bad=deepcopy(rows);bad[0]['candidate']['split']='dev'
        with self.assertRaises(ValueError):list(finite_mixture_schedule(bad,{'t':[0],'u':[1]}))

    def test_planner_rehearsal_is_explicit_and_pool_separated(self):
        rows=[dict(candidate=dict(split='train',source_group=f't:{i}'),approval=dict(pool='planner',event_id=str(i))) for i in range(3)]
        with self.assertRaises(ValueError):list(finite_mixture_schedule(rows,{'t':[0],'u':[1]},batch_size=8))
        batches=list(finite_mixture_schedule(rows,{'t':[0],'u':[1]},batch_size=8,pool='planner',maximum_event_passes=1))
        self.assertEqual([b['new_count'] for b in batches],[2,1])
        self.assertTrue(all(b['expert_count']>b['new_count'] for b in batches))


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_coverage import build_catalog,skill_family,select_grasp_sources
from recovery_corpus import group_key,split_group


def episode(instance,split='train'):
    def segment(verb,start,end,bundle=1):
        return dict(start=start,end=end,parent='task',semantic=json.dumps([dict(verb=verb,target='object',unbound_relation='')]*bundle))
    return dict(task_name='task',split=split,row=dict(task_index=0,task_instance_id=instance,episode_index=instance),
        segments=[segment('GRASP',0,16,2),segment('GRASP',16,48),segment('GRASP',50,80),segment('OPEN_DOOR',80,96)])


class CoverageTests(unittest.TestCase):
    def test_protected_groups_and_parallel_members_do_not_inflate_events(self):
        out=build_catalog([episode(1),episode(2),episode(3,'eval')],{'task:2'})
        self.assertEqual(out['source_groups'],1);self.assertEqual(len(out['queue']),2)
        row=next(r for r in out['queue'] if r['verb']=='GRASP')
        self.assertEqual(row['bundle_size'],1);self.assertEqual(row['segment_start'],16)
        self.assertFalse(out['physical_recovery_verified']);self.assertFalse(out['training_approved'])

    def test_duplicate_sources_and_task_ids_reject(self):
        with self.assertRaises(ValueError):build_catalog([episode(1),episode(1)],set())
        changed=episode(2);changed['row']['task_index']=7
        with self.assertRaises(ValueError):build_catalog([episode(1),changed],set())
        self.assertEqual(skill_family('PLACE_IN'),'placement_release')
        self.assertEqual(skill_family('OPEN_DOOR'),'articulation')

    def test_task_balancing_preserves_holdouts_and_first_attempts(self):
        rows=[]
        for task in ('task','other'):
            for i in range(100):
                ep=episode(i);ep['task_name']=task;rows.append(ep)
        prior={group_key('task',i) for i in range(3)}
        selected,coverage=select_grasp_sources(rows,{'task:4'},['task','other','missing'],dict(train=4,dev=3),prior)
        identities=[group_key(ep['task_name'],ep['row']['task_instance_id']) for ep,_,_ in selected]
        self.assertTrue(prior.isdisjoint(identities));self.assertNotIn('task:4',identities)
        self.assertEqual([ep['task_name'] for ep,_,_ in selected[:2]],['task','other'])
        self.assertEqual(len(identities),len(set(identities)))
        for ep,seg,split in selected:
            self.assertEqual(len(json.loads(seg['semantic'])),1)
            self.assertEqual(split,split_group(ep['task_name'],ep['row']['task_instance_id']))
        self.assertEqual(sum(r['missing'] for r in coverage if r['task']=='missing'),7)

    def test_no_bound_grasp_is_a_gap_not_a_fabricated_source(self):
        ep=episode(1)
        for seg in ep['segments']:
            skills=json.loads(seg['semantic'])
            for skill in skills:skill['unbound_relation']='not bound'
            seg['semantic']=json.dumps(skills)
        selected,coverage=select_grasp_sources([ep],set(),['task'],dict(train=2,dev=1))
        self.assertEqual(selected,[]);self.assertEqual(sum(r['missing'] for r in coverage),3)


if __name__=='__main__':unittest.main()

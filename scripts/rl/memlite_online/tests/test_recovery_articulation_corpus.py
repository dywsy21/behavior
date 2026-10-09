from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_articulation_corpus import CLEAN, validate_articulation_branch
from recovery_articulation_teacher import functional_goal
from recovery_corpus import digest
from recovery_observer_training import request_key


def fixture():
    plans=[]
    for tick, decision in ((0, 'EXECUTE'), (28, 'RETRY')):
        p=dict(control_step=tick, decision=decision, parent_goal='open washer',
               active_skills_semantic_json='[{"verb":"OPEN_DOOR","target":"washer_1"}]')
        p['event_sha256']=digest(p);plans.append(p)
    def physical(fraction):
        return dict(target_name='washer_1', entity='washer.n.01_1', open=fraction>.05,
                    directed_open_fraction=fraction, goal_predicate=functional_goal(fraction>.05,fraction,True))
    seed=dict(physical=physical(.4));rows=[];before=seed['physical']
    for t in range(52):
        phase='clean_hold' if t<16 else 'fault_reverse' if t<28 else 'corrective'
        after=physical(.4 if t<16 else .08 if t<28 else min(.4,.09+.04*(t-28)))
        p=plans[int(t>=28)]
        rows.append(dict(control_step=t,proprio_before=[t]*61,proprio_after=[t+1]*61,
            action_executed_raw23=[0.]*23,simulator_apply_ack=True,
            label_kind='injected_fault_not_BC' if phase=='fault_reverse' else CLEAN,
            teacher=dict(phase=phase),physical_before=before,physical_audit=after,
            context=dict(context_id=p['event_sha256'],parent_goal=p['parent_goal'],
                         active_skills_semantic_json=p['active_skills_semantic_json'])))
        before=after
    return rows,dict(controls=len(rows),verb='OPEN_DOOR'),plans,seed


class ArticulationCorpusTests(unittest.TestCase):
    def test_retry_frame_old_attempt_failed_new_attempt_unknown(self):
        rows,m,p,s=fixture();labels,prior=validate_articulation_branch(rows,m,p,s)
        self.assertEqual(labels[16],'SUCCEEDED')
        self.assertEqual(labels[27],'UNKNOWN')
        self.assertEqual(labels[28],'UNKNOWN')
        self.assertEqual(prior[28],dict(value='FAILED',context_id=p[0]['event_sha256'],evidence_end_control_step=28))
        self.assertEqual(labels[29],'IN_PROGRESS')
        self.assertEqual(labels[51],'SUCCEEDED')
        original=dict(labels)
        # Future outcomes cannot alter the predecision failure at control 28.
        for row in rows[36:]:
            row['physical_before']=deepcopy(rows[35]['physical_audit'])
            row['physical_audit']=deepcopy(rows[35]['physical_audit'])
        changed,other=validate_articulation_branch(rows,m,p,s)
        self.assertEqual({k:v for k,v in original.items() if k<=28},{k:v for k,v in changed.items() if k<=28})
        self.assertEqual(other,prior)

    def test_wrong_target_unacknowledged_fault_and_context_rejected(self):
        for field in ('target','ack','fault','context','predicate'):
            rows,m,p,s=fixture()
            if field=='target':rows[20]['physical_audit']['target_name']='other'
            if field=='ack':rows[20]['simulator_apply_ack']=False
            if field=='fault':rows[20]['label_kind']=CLEAN
            if field=='context':rows[28]['context']['context_id']=p[0]['event_sha256']
            if field=='predicate':rows[20]['physical_audit']['goal_predicate']=True
            with self.assertRaises(ValueError):validate_articulation_branch(rows,m,p,s)

    def test_features_bind_the_approved_attempt(self):
        row=dict(candidate=dict(sample_id='image'),approval=dict(label=dict(member_index=0)))
        old=request_key(row)
        row['approval']['label']['history_role']='predecision'
        self.assertNotEqual(old,request_key(row))
        self.assertEqual(request_key(row),digest(['image','predecision',0]))


if __name__=='__main__':unittest.main()

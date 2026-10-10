from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).resolve().parents[4]/'src')]
from recovery_postfit_feedback import clock_requests,predicted_execution_feedback,make_provenance,validate_provenance
from test_recovery_postfit import PostfitTests


class PostfitFeedbackTests(unittest.TestCase):
    def fixture(self,end=64):
        from g05.utils.memlite_skill_protocol import canonical_json
        bundle=canonical_json([dict(verb='GRASP',target='radio_1',source='table',destination='',
            target_part='',arm='RIGHT',unbound_relation='')])
        memory=canonical_json(dict(issued_command_history=[],task_name='turn on radio',verified_world_facts=[]))
        rows=[];history=[]
        for t in range(0,end+1,4):
            row=dict(sample_id='s'+str(t),source_episode=['run','ep'],source_group='turn on radio:1',
                control_step=t,split='train',task='turn_on_radio',instance_id=1)
            ctx=dict(control_step=0,observation_control_step=t,served_controls=t,history_is_partial=False,
                intent_started_control_step=0,attempt_number=1,repeated_planning_count=0,context_id='initial',
                issued_skills_semantic_json=bundle,parent_goal='Task goal: turn on radio',memory=memory)
            h=dict(sample_id=row['sample_id'],source_episode=row['source_episode'],source_group=row['source_group'],
                control_step=t,observable=deepcopy(ctx),predecision=deepcopy(ctx) if t==end else None)
            if t==end:h['observable'].update(intent_started_control_step=end,attempt_number=2,context_id='retry')
            rows.append(row);history.append(h)
        return rows,history,[rows[-1]],PostfitTests().exposure()

    def trace(self,end=64):
        a,h,s,e=self.fixture(end);r=clock_requests(a,h,s,e)
        return a,h,s,e,[dict(x,logits=[-8.,-8.,8.,-8.]) for x in r]

    def test_two_independently_complete_windows(self):
        _,_,_,_,r=self.trace(64)
        self.assertEqual([x['control_step'] for x in r],[0,16,32,48,64])
        self.assertEqual([c['control_step'] for c in r[-2]['checks']],[0,16,32,48])
        self.assertEqual([c['control_step'] for c in r[-1]['checks']],[16,32,48,64])

    def test_partial_clock_is_stale_unknown(self):
        _,h,s,_,r=self.trace(24)
        self.assertEqual([x['control_step'] for x in r],[0,16])
        out=json.loads(predicted_execution_feedback(s[0],h[-1],r,PostfitTests().calibration()))
        self.assertEqual(out['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual(out['estimated_member_outcomes'][0]['confidence'],0.)
        self.assertEqual(out['same_intent_controls'],24)

    def test_fresh_two_agreed_checks_use_estimate_not_truth(self):
        _,h,s,_,r=self.trace(32)
        out=json.loads(predicted_execution_feedback(s[0],h[-1],r,PostfitTests().calibration()))
        self.assertEqual(out['estimated_bundle_outcome'],'FAILED')
        r[-1]['logits']=[-8.,8.,-8.,-8.]
        self.assertEqual(json.loads(predicted_execution_feedback(s[0],h[-1],r,
            PostfitTests().calibration()))['estimated_bundle_outcome'],'UNKNOWN')

    def test_isolation_and_missing_real_frames_fail(self):
        for field,value in [('source_group','other:1'),('source_episode',['run','other']),('task','other'),('split','dev')]:
            a,h,s,e=self.fixture();a[4][field]=value
            with self.assertRaises(ValueError):clock_requests(a,h,s,e)
        a,h,s,e=self.fixture();a=[x for x in a if x['control_step']!=16]
        with self.assertRaises(ValueError):clock_requests(a,h,s,e)
        a,h,s,e=self.fixture();h[4]['observable']['intent_started_control_step']=16
        with self.assertRaises(ValueError):clock_requests(a,h,s,e)

    def test_predictions_never_reuse_future_or_other_episode(self):
        for mutation in ('duplicate','episode','window','nan'):
            _,h,s,_,r=self.trace(64)
            if mutation=='duplicate':r.append(r[-1])
            if mutation=='episode':r[-1]['source_episode']=['run','other']
            if mutation=='window':r[-2]['checks']=r[-2]['checks'][1:]
            if mutation=='nan':r[-1]['logits'][0]=float('nan')
            with self.assertRaises(ValueError):predicted_execution_feedback(s[0],h[-1],r,PostfitTests().calibration())

    def test_explicit_separate_models_and_all_exposures(self):
        e=PostfitTests().exposure()
        p=make_provenance(e,high_sha256='c'*64,exposure_sha256='d'*64,calibration_sha256='e'*64,
            trace_sha256='f'*64,target_groups=['new:1'])
        self.assertTrue(validate_provenance(p,group='new:1',high_sha256='c'*64))
        with self.assertRaises(ValueError):validate_provenance(p,group='new:1',high_sha256='a'*64)
        for g in e['excluded_groups']:
            bad=deepcopy(p);bad['target_groups'].append(g)
            with self.assertRaises(ValueError):validate_provenance(bad,group='new:1',high_sha256='c'*64)


if __name__=='__main__':unittest.main()

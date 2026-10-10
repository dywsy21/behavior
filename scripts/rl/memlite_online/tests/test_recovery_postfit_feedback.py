from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).resolve().parents[4]/'src')]
from recovery_postfit_feedback import clock_requests,predicted_execution_feedback,make_provenance,validate_provenance,validate_training_release,STATUS
from recovery_corpus import digest,file_sha
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

    def test_training_recomputes_actual_logits_and_exact_windows(self):
        a,h,s,e,trace=self.trace(64);cal=PostfitTests().calibration()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'audit').mkdir();(p/'history').mkdir();(p/'admission').mkdir()
            def write(name,obj,lines=False):
                path=p/name;path.write_text((''.join(json.dumps(x)+'\n' for x in obj)) if lines else json.dumps(obj));return path
            paths={}
            paths['admission']=write('admission/admission.json',{})
            paths['inventory']=write('audit/inventory.json',[])
            anchors=write('audit/anchors.jsonl',a,True);paths['history']=write('history/contexts.jsonl',h,True)
            write('history/receipt.json',dict(inventory_sha256=digest([]),anchors_sha256=file_sha(anchors),contexts_sha256=file_sha(paths['history'])))
            paths['observer_calibration']=write('calibration.json',cal)
            e['bindings']={'calibration':dict(sha256=file_sha(paths['observer_calibration']))}
            paths['observer_exposure']=write('exposure.json',e)
            paths['observer_prediction_trace']=write('trace.json',trace)
            provenance=make_provenance(e,high_sha256='c'*64,exposure_sha256=file_sha(paths['observer_exposure']),
                calibration_sha256=file_sha(paths['observer_calibration']),trace_sha256=file_sha(paths['observer_prediction_trace']),
                target_groups=[s[0]['source_group']])
            feedback=[dict(sample_id=s[0]['sample_id'],source_group=s[0]['source_group'],control_step=64,provenance=provenance,
                execution_feedback=predicted_execution_feedback(s[0],h[-1],trace,cal))]
            paths['feedback']=write('feedback.jsonl',feedback,True)
            result=dict(status=STATUS,diagnostic_only=False,optimizer_updates=0,calibration_refit=False,observer_weights_updated=False,
                all_admitted_rows_retained=True,frozen_before_sha256='1'*64,frozen_after_sha256='1'*64,
                high_sha256='c'*64,planner_high_sha256='c'*64,admission_sha256=file_sha(paths['admission']),
                feedback_sha256=file_sha(paths['feedback']),exposure_sha256=file_sha(paths['observer_exposure']),
                calibration_sha256=file_sha(paths['observer_calibration']),prediction_trace_sha256=file_sha(paths['observer_prediction_trace']),
                requests_sha256=digest([{k:v for k,v in r.items() if k!='logits'} for r in trace]),rows=1)
            paths['feedback_receipt']=write('feedback_receipt.json',result)
            ticket=dict(admission=str(p/'admission'),files={k:dict(path=str(v),sha256=file_sha(v)) for k,v in paths.items()})
            recipe=dict(parents=dict(high=dict(sha256='c'*64)),H1=dict(feedback='frozen_source_disjoint_observer_v1',
                observer_backbone_sha256='a'*64,observer_sha256='b'*64,exposure=dict(sha256=file_sha(paths['observer_exposure'])),
                calibration=dict(sha256=file_sha(paths['observer_calibration']))))
            with patch('recovery_sft_data.require_training_pool',return_value=({},[dict(candidate=s[0])])):
                self.assertEqual(validate_training_release(ticket,recipe)['real_checks'],5)
                # Even if an attacker rehashes the receipt and feedback, logits
                # must actually imply that text under the serving contract.
                feedback[0]['execution_feedback']=feedback[0]['execution_feedback'].replace('FAILED','SUCCEEDED')
                write('feedback.jsonl',feedback,True);result['feedback_sha256']=file_sha(paths['feedback'])
                write('feedback_receipt.json',result)
                for k in ('feedback','feedback_receipt'):ticket['files'][k]['sha256']=file_sha(paths[k])
                with self.assertRaisesRegex(ValueError,'actual fixed logits'):validate_training_release(ticket,recipe)


if __name__=='__main__':unittest.main()

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import canonical,group_key
from recovery_features import feature_requests,balanced_event_schedule,group_folds,validate_prediction_provenance,require_history_protocol
from recovery_history import join_causal_history
from recovery_observer_training import temporal_batch,calibrate,predicted_feedback,feedback_text
from recovery_planner_data import planner_projection
from recovery_train_contract import validate_launch,validate_h0_event_budget
from test_recovery_history import fixture

helpers=Path(__file__).resolve().parents[4]/'src/g05/models/g05/helpers'
package=types.ModuleType('_pipeline_test_helpers');package.__path__=[str(helpers)];sys.modules[package.__name__]=package
spec=importlib.util.spec_from_file_location(package.__name__+'.temporal_outcome',helpers/'temporal_outcome.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def joined():
    ep,logs=fixture();anchors=[];raw={}
    for t in range(0,273,16):
        log=logs[min(t//128,2)]
        anchors.append(dict(sample_id=str(t),source_episode=['run','episode'],task=ep['task'],
            source_group=group_key(ep['task'],211),split='train',control_step=t,
            actor_input=dict(issued_skills_semantic_json=log['event']['active_skills_semantic_json'],
                             parent_goal=log['event']['parent_goal'])))
        raw[t]=dict(context=dict(context_id=log['context_id']))
    class Reader:
        def episode(self,_): return raw,{}
    return anchors,join_causal_history(anchors,[dict(episode=ep)],logs,Reader())


def provenance(group):
    return dict(high_sha256='a'*64,observer_sha256='b'*64,trained_groups=['train-a','train-b'],
        calibration_groups=['cal-a','cal-b'],target_groups=[group],feature_cache_sha256='c'*64,
        fold=1,calibrated=False,temperature=1.,calibration_receipt_sha256='d'*64)


class PipelineTests(unittest.TestCase):
    def test_optional_absolute_proprio_is_observable_and_gradient_insulated(self):
        from copy import deepcopy
        a=dict(context=torch.randn(1,3,16),proprio=torch.zeros(1,3,27),
            steps=torch.tensor([[0,16,32]]),valid=torch.ones(1,3,dtype=torch.bool))
        b=deepcopy(a);b['proprio']+=.5
        for absolute in (False,True):
            head=module.TemporalOutcomeObserver(16,width=8,include_absolute_proprio=absolute)
            torch.nn.init.normal_(head.residual.weight,std=.2);head.eval()
            self.assertEqual(torch.allclose(head(**a),head(**b)),not absolute)
            a['proprio'].requires_grad_(True)
            head(**a).sum().backward();self.assertIsNone(a['proprio'].grad)

    def test_preparation_ticket_cannot_start_formal_or_expand_smoke(self):
        from recovery_corpus import file_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=root/'recipe.json';path=root/'ticket.json'
            recipe.write_text(json.dumps(dict(maximum_updates_per_line=1000,maximum_event_passes=5,
                                               maximum_wall_seconds_per_line=14400)))
            value=dict(schema='recovery_sft_launch_v1',component='L0',formal_training_authorized=False,
                engineering_smoke=True,maximum_updates=2,event_passes=1,wall_seconds=1800,world_size=8,micro_batch=1,global_batch=16,
                files=dict(recipe=dict(path=str(recipe),sha256=file_sha(recipe))))
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'not authorized'): validate_launch(path,'L0')
            validate_launch(path,'L0',engineering=True)
            value['maximum_updates']=3;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'two original-expert'): validate_launch(path,'L0',engineering=True)
            recipe.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed pinned'): validate_launch(path,'L0',engineering=True)

    def test_h0_all_folds_obey_explicit_smaller_event_budget(self):
        validate_h0_event_budget({'event-a':5,'event-b':4},dict(event_passes=5))
        for counts,passes in [({'event-a':5},3),({'event-a':6},5),({},5)]:
            with self.assertRaises(ValueError):validate_h0_event_budget(counts,dict(event_passes=passes))

    def test_causal_feature_clocks_and_h1_self_target_exclusion(self):
        anchors,history=joined()
        req=feature_requests(anchors,history,[('256','predecision',0)])[0]
        self.assertEqual([c['control_step'] for c in req['checks']],[208,224,240,256])
        self.assertEqual(req['checks'][-1]['memory'],history[8]['observable']['memory'])
        self.assertNotIn('label',canonical(req['checks']))
        with self.assertRaises(ValueError): feature_requests(anchors,history,[('0','predecision',0)])
        with self.assertRaises(ValueError): feature_requests(anchors,history,[('256','predecision',1)])
        bad=deepcopy(history);bad[16]['predecision']['served_controls']=12
        with self.assertRaises(ValueError): feature_requests(anchors,bad,[('256','predecision',0)])

    def test_short_attempt_has_multiple_distinct_real_checks(self):
        anchors,history=joined()
        req=feature_requests(anchors,history,[('32','observable',0)])[0]
        self.assertEqual([c['control_step'] for c in req['checks']],[0,16,32])
        self.assertTrue(all(c['control_step']<=32 for c in req['checks']))

    def test_first_short_interval_retains_actual_issuance_observation(self):
        anchors,history=joined()
        current=deepcopy(anchors[1]);current.update(sample_id='8',control_step=8)
        ctx=deepcopy(history[1]);ctx.update(sample_id='8',control_step=8)
        ctx['observable'].update(observation_control_step=8,served_controls=8)
        anchors.append(current);history.append(ctx)
        old=feature_requests(anchors,history,[('8','observable',0)])[0]
        self.assertEqual([c['control_step'] for c in old['checks']],[8])
        self.assertNotIn('history_protocol',old)
        new=feature_requests(anchors,history,[('8','observable',0)],history_protocol='attempt_start_short_v1')[0]
        self.assertEqual([c['control_step'] for c in new['checks']],[0,8])
        self.assertEqual(new['checks'][0]['served_controls'],0)
        self.assertEqual(new['checks'][-1],old['checks'][-1])
        # Missing or another attempt's issuance is never substituted or copied.
        for bad in ('missing','intent','group','partial'):
            aa,hh=deepcopy(anchors),deepcopy(history)
            if bad=='missing':aa=aa[1:];hh=hh[1:]
            if bad=='intent':hh[0]['observable']['intent_started_control_step']=1
            if bad=='partial':hh[0]['observable']['history_is_partial']=True
            if bad=='group':
                aa[0]['source_group']='another-source'
                with self.assertRaisesRegex(ValueError,'Cross-group'):
                    feature_requests(aa,hh,[('8','observable',0)],history_protocol='attempt_start_short_v1')
                continue
            rr=feature_requests(aa,hh,[('8','observable',0)],history_protocol='attempt_start_short_v1')[0]
            self.assertEqual([c['control_step'] for c in rr['checks']],[8])
        for sid in ('0','32','256'):
            role='predecision' if sid=='256' else 'observable'
            before=feature_requests(anchors,history,[(sid,role,0)])[0]
            after=feature_requests(anchors,history,[(sid,role,0)],history_protocol='attempt_start_short_v1')[0]
            self.assertEqual(before['checks'],after['checks'])

    def test_history_protocol_is_bound_to_config_cache_receipt_and_requests(self):
        self.assertEqual(require_history_protocol({}, {}, {'requests':[{}]}),'cadence16_v1')
        value=dict(history_protocol='attempt_start_short_v1')
        cache=dict(value,requests=[dict(value)])
        self.assertEqual(require_history_protocol(value,value,cache),'attempt_start_short_v1')
        for config,receipt,data in (({},value,cache),(value,{},cache),(value,value,{'requests':[value]}),
                                    (value,value,dict(value,requests=[{}]))):
            with self.assertRaisesRegex(ValueError,'history protocol'):
                require_history_protocol(config,receipt,data)

    def test_group_folds_and_calibration_group_leak_rejected(self):
        fold=group_folds([str(i) for i in range(7)])
        self.assertEqual(set(fold.values()),{0,1,2})
        self.assertEqual(fold,group_folds([str(i) for i in reversed(range(7))]))
        with self.assertRaises(ValueError): group_folds(['a','b'])
        row=provenance('target');self.assertTrue(validate_prediction_provenance(row,group='target',high_sha256='a'*64))
        for key in ('trained_groups','calibration_groups'):
            bad=deepcopy(row);bad[key].append('target')
            with self.assertRaises(ValueError): validate_prediction_provenance(bad,group='target',high_sha256='a'*64)

    def test_event_schedule_not_frame_inflated(self):
        rows=[dict(candidate=dict(split='train',task='t'+str(i%2),source_group='g'+str(i)),
                   approval=dict(event_id='e'+str(i))) for i in range(6)]
        repeated=rows+[deepcopy(rows[0]) for _ in range(50)]
        batches=list(balanced_event_schedule(repeated,batch_size=4,passes=3))
        self.assertEqual(sum(len(b['rows']) for b in batches),18)
        self.assertEqual([len(b['rows']) for b in batches],[4,2,4,2,4,2])

    def test_event_schedule_visits_known_classes_without_replacement(self):
        rows=[dict(candidate=dict(split='train',task='t',source_group='g'),
            approval=dict(event_id='e',label=dict(value=value))) for value in
            ['FAILED','IN_PROGRESS']+['SUCCEEDED']*40]
        batches=list(balanced_event_schedule(rows,batch_size=4,passes=3))
        self.assertEqual({rows[b['rows'][0][1]]['approval']['label']['value'] for b in batches},
                         {'FAILED','IN_PROGRESS','SUCCEEDED'})
        self.assertTrue(all(len(b['rows'])==1 for b in batches))

    def test_tiny_perfect_calibration_still_not_ready(self):
        labels=torch.tensor([0,1,2]);logits=torch.full((3,4),-8.);logits[range(3),labels]=8
        result=calibrate(logits,labels,['a','b','c'])
        self.assertFalse(result['ready']);self.assertGreater(len(result['blockers']),0)
        many=labels.repeat(40);perfect=torch.full((120,4),-8.);perfect[range(120),many]=8
        groups=[f'independent:{i}' for i in range(120)]
        good=calibrate(perfect,many,groups)
        self.assertTrue(good['ready'])
        wrong=calibrate(perfect.roll(1,1),many,groups)
        self.assertFalse(wrong['ready'])
        with self.assertRaisesRegex(ValueError,'independent source group'):
            calibrate(perfect,many,['a','b']*60)
        with self.assertRaises(ValueError):calibrate(logits,labels,['a','b'])

    def test_temporal_head_real_backward_and_incomplete_history_unknown(self):
        head=module.TemporalOutcomeObserver(16,width=8)
        f=dict(context=torch.randn(3,16),proprio=torch.randn(3,27),steps=torch.tensor([0,128,256]))
        batch=temporal_batch([f,{k:v[:1] for k,v in f.items()}],'cpu')
        loss=torch.nn.functional.cross_entropy(head(**batch),torch.tensor([0,1]));loss.backward()
        self.assertEqual(sum(p.grad is not None for p in head.parameters()),14)
        request=dict(request_id='r',member_index=0)
        pred=predicted_feedback(head,request,{'r':{k:v[:1] for k,v in f.items()}},None,
            dict(ready=True,temperature=1.,minimum_confidence=0.),'cpu')
        self.assertEqual(pred['estimated_outcome'],'UNKNOWN')

    def test_planner_requires_actual_target_causal_inputs_and_oof(self):
        anchors,history=joined();row=anchors[8];h=history[8];current=h['observable'];prior=h['predecision']
        target=dict(schema='recovery_verified_plan_v1',sample_id=row['sample_id'],source_episode=row['source_episode'],
            control_step=128,parent_goal=current['parent_goal'],active_skills_semantic_json=current['issued_skills_semantic_json'],
            decision=current['issued_decision'],memory_update=current['memory'])
        pred=dict(sample_id=row['sample_id'],source_group=row['source_group'],control_step=128,
            provenance=provenance(row['source_group']),execution_feedback=feedback_text(prior,[dict(member=0,estimated_outcome='UNKNOWN',confidence=0.)]))
        projection=planner_projection(row,h,target,pred,'a'*64)
        self.assertEqual(projection['memory'],prior['memory']);self.assertFalse(projection['outcome_supervision_mask'])
        self.assertEqual(projection['outcome_target'],'UNKNOWN');self.assertFalse(projection['low_action_supervision_mask'])
        calibrated=deepcopy(pred);calibrated['provenance']['calibrated']=True
        calibrated['execution_feedback']=feedback_text(prior,[dict(member=0,estimated_outcome='FAILED',confidence=.97)])
        projected=planner_projection(row,h,target,calibrated,'a'*64)
        self.assertEqual(projected['known_previous_outcome'],'UNKNOWN')
        self.assertEqual(json.loads(projected['execution_feedback'])['estimated_bundle_outcome'],'FAILED')
        self.assertEqual(projected['outcome_target'],'UNKNOWN')
        bad=deepcopy(target);bad['decision']='RETRY'
        with self.assertRaises(ValueError): planner_projection(row,h,bad,pred,'a'*64)
        bad=deepcopy(pred);bad['provenance']['trained_groups'].append(row['source_group'])
        with self.assertRaises(ValueError): planner_projection(row,h,target,bad,'a'*64)
        bad=deepcopy(pred);v=json.loads(bad['execution_feedback']);v['physical_truth']='SUCCEEDED';bad['execution_feedback']=json.dumps(v)
        with self.assertRaises(ValueError): planner_projection(row,h,target,bad,'a'*64)


if __name__=='__main__':unittest.main()

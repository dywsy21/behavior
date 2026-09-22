import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/vlm_sft'), str(ROOT/'src')]
import native_completion_saved_eval as module
from native_completion_serve import DecisionEngine
from native_completion_protocol import VARIANTS, snapshot_sha
from test_native_completion_training import fixture, prediction


class SavedCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        actor, raw = fixture(); directory = Path(self.temp.name)
        for view, data in raw.items(): (directory/(view+'.png')).write_bytes(data)
        self.cases = [{'id':f'state_{i:02d}', 'actor':copy.deepcopy(actor),
            'images':{v:str(directory/(v+'.png')) for v in raw},
            'expected_status':'REQUEST_VERIFY' if i in (5,7) else 'CONTINUE',
            'saved_public_holding_verified':i in (5,6,7),
            'private_same_tick_score':{'held':True, 'target_pose':'NEVER_SEND'},
            'clock':{'prefix_control':1038, 'native_control':i}} for i in range(8)]
        self.identity = {'max_calls':64, 'variants':list(VARIANTS),
            'instructions':[actor['active_instruction']], 'adapter_sha256':{v:'a'*64 for v in VARIANTS}}
        self.seen=[]; self.ledger=[]
        def backend(variant,row,images):
            self.seen.append((variant,copy.deepcopy(row),{k:v.tobytes() for k,v in images.items()}))
            return prediction('CONTINUE' if row['query_kind']=='status' else 'RIGHT_UP')
        self.engine=DecisionEngine(self.identity,'b'*64,backend,check=lambda:None,record=lambda _:None)

    def evaluate(self,**kwargs):
        deadline=kwargs.pop('deadline') if 'deadline' in kwargs else time.monotonic()+60
        return module.evaluate(kwargs.pop('cases',self.cases),self.identity,'b'*64,
            call=kwargs.pop('call',lambda value,remaining:self.engine.decide(value)),
            check=kwargs.pop('check',lambda:None), record=self.ledger.append,
            deadline=deadline,initial_calls=kwargs.pop('initial_calls',0),**kwargs)

    def test_real_transaction_projection_same_pixels_and_no_private_fields(self):
        result=self.evaluate();self.assertEqual(result['actual_queries'],32)
        self.assertEqual(len(result['decisions']),16)
        for i in range(0,len(self.seen),4):
            group=self.seen[i:i+4]
            self.assertEqual(group[0][1]['actor'],group[3][1]['actor'])
            self.assertEqual(group[0][2],group[3][2])
        for _,row,_ in self.seen:
            serialized=json.dumps(row)
            for forbidden in ('NEVER_SEND','private_same_tick','native_control','expected_status','eval_t1'):
                self.assertNotIn(forbidden,serialized)
            self.assertEqual(len(row['actor']),6)
        self.assertIsNone(result['new_simulator_success_rate'])

    def test_request_only_has_no_motion_and_counts_one_actual_query(self):
        self.engine.predict=lambda *args:prediction('REQUEST_VERIFY')
        result=self.evaluate()
        self.assertEqual(result['actual_queries'],16)
        self.assertTrue(all(d['response']['motion'] is None for d in result['decisions']))
        self.assertEqual(result['confusion'][VARIANTS[0]]['false_early_request'],6)

    def test_confusion_baselines_and_public_holding_is_not_completion(self):
        request=module.confusion(self.cases,['REQUEST_VERIFY']*8)
        self.assertEqual(request['true_request'],2);self.assertEqual(request['false_early_request'],6)
        self.assertEqual(request['public_verified_but_strictly_early_request'],1)
        never=module.confusion(self.cases,['CONTINUE']*8)
        self.assertEqual(never['missed_late_request'],2);self.assertEqual(never['true_continue'],6)
        unknown=copy.deepcopy(self.cases);unknown[5]['expected_status']='UNKNOWN'
        self.assertEqual(module.confusion(unknown,['REQUEST_VERIFY']*8)['unknown_label'],1)

    def test_missing_exact_outcome_never_borrows_later_success(self):
        score={10:{'left':{'outcome':'IN_PROGRESS'},'right':{'outcome':'SUCCEEDED'}}}
        self.assertEqual(module.expected_status(score,9),'UNKNOWN')
        self.assertEqual(module.expected_status(score,10),'REQUEST_VERIFY')
        score[11]={'left':{'outcome':'UNKNOWN'},'right':{'outcome':'IN_PROGRESS'}}
        self.assertEqual(module.expected_status(score,11),'UNKNOWN')

    def test_error_or_changed_call_counter_stops_without_retry(self):
        calls=[]
        def fail(request,remaining):calls.append(request);raise TimeoutError('bounded transport')
        with self.assertRaises(TimeoutError):self.evaluate(call=fail)
        self.assertEqual(len(calls),1)
        self.engine.calls=1
        with self.assertRaisesRegex(ValueError,'Concurrent'):self.evaluate()
        self.assertEqual(self.engine.calls,3)

    def test_deadline_blocks_new_request_and_rejects_late_result(self):
        with self.assertRaises(TimeoutError):self.evaluate(deadline=time.monotonic()-1)
        self.assertEqual(self.engine.calls,0)
        with patch.object(module.time,'monotonic',side_effect=[0,0,2]):
            with self.assertRaises(TimeoutError):self.evaluate(deadline=1)
        self.assertEqual(self.engine.calls,2)

    def test_wrong_pixels_and_actor_privilege_rejected_before_model(self):
        self.cases[0]['actor']['held']=True
        with self.assertRaises(ValueError):self.evaluate()
        self.assertEqual(self.engine.calls,0)
        del self.cases[0]['actor']['held']
        Path(self.cases[0]['images']['head']).write_bytes(b'wrong')
        with self.assertRaises(ValueError):self.evaluate()
        self.assertEqual(self.engine.calls,0)

    def test_case_and_total_call_caps_and_health_identity(self):
        for cases in (self.cases[:7],self.cases+[self.cases[0]],[self.cases[0]]*8):
            with self.assertRaises(ValueError):self.evaluate(cases=cases)
        for counter in (True,33,-1,0.):
            with self.assertRaises(ValueError):self.evaluate(initial_calls=counter)
        health={**self.identity,'identity_sha256':'b'*64,'calls':0}
        module.checked_health(health,self.identity,'b'*64,0)
        for bad in ({**health,'calls':1},{**health,'identity_sha256':'c'*64}):
            with self.assertRaises(ValueError):module.checked_health(bad,self.identity,'b'*64,0)

    def test_registered_config_and_exact_source_lists(self):
        self.assertEqual(json.loads((ROOT/'configs/vlm_sft/h09ac_saved_completion_eval.json').read_text()),module.CONFIG)
        self.assertEqual(len(module.SELECTION),8)
        self.assertEqual(module.SERVICE_CODE,'9deafb1b65cc9e0c3c91371af0eb2f815b063774')
        self.assertEqual(module.CONFIG['max_queries'],32)
        with self.assertRaises(FileNotFoundError):module.checked_inventory(Path(self.temp.name),next(iter(module.RUNS)))


if __name__=='__main__':unittest.main()

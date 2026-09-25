from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/vlm_sft'),str(REPO/'src')]
import calibrate_grounding_teacher as worker
import launch_grounding_calibration as launch
from visual_grounding import messages,decode_response


class FixtureTokenizer:
    eos_token_id=1;pad_token_id=0;unk_token_id=2;all_special_ids=[0,1,2]
    def __len__(self):return 1000
    def convert_tokens_to_ids(self,s):return 1
    def convert_ids_to_tokens(self,i):return '<|im_end|>'
    def decode(self,ids,skip_special_tokens=False):return ''.join('<special>' if i<3 else chr(i-10) for i in ids)


class GroundingCalibrationTests(unittest.TestCase):
    def test_only_train_source_selection_and_fixed_repeat(self):
        rows=[{'id':str(i),'task':i//36,'instance':i//9,'split':'visual_train' if i<81 else 'visual_validation'} for i in range(108)]
        with patch.object(worker,'load_rows',return_value=(rows,{'data':'fixed'})):
            train,repeat,data=worker.training_rows();self.assertEqual(len(train),81);self.assertEqual(len(repeat),16)
            self.assertTrue({r['id'] for r in repeat}<={r['id'] for r in train})
            self.assertTrue(all(r['split']=='visual_train' for r in train))
            self.assertEqual((train,repeat,data),worker.training_rows())

    def test_invalid_format_counts_as_wrong_not_dropped(self):
        rows=[{'id':'a','label':'P'},{'id':'b','label':'N'},{'id':'c','label':'U'}]
        predictions=[{'id':'a','format_valid':True,'parsed':{'visibility':'present'}},
                     {'id':'b','format_valid':False,'parsed':None},
                     {'id':'c','format_valid':True,'parsed':{'visibility':'absent'}}]
        result=worker.report(predictions,rows)
        self.assertEqual(result['accuracy_including_invalid'],1/3)
        self.assertEqual(result['format_failures'],['b']);self.assertFalse(result['training_label_release'])
        self.assertFalse(result['bbox_gold_available'])
        self.assertEqual(result['confusion']['N']['invalid'],1)
        self.assertEqual(result['confusion']['U']['N'],1)

    def test_gpu_ownership_cap_headroom_and_release(self):
        baseline={u:{'processes':[],'used_mib':0,'free_mib':81152} for u in launch.GPU_UUIDS}
        baseline[launch.GPU_UUIDS[0]]['processes']=[{'pid':123,'used_mib':12548}]
        launch.check_resources(baseline)
        current=deepcopy(baseline);current[launch.GPU_UUID]['processes']=[{'pid':456,'used_mib':62000}]
        with patch.object(launch,'belongs_to_session',side_effect=lambda pid,sid:pid==sid):
            launch.check_resources(current,baseline,456)
            with self.assertRaises(RuntimeError):launch.check_resources(current,baseline,456,released=True)
            with self.assertRaises(RuntimeError):launch.check_resources(current,baseline,999)
            current[launch.GPU_UUID]['processes'][0]['used_mib']=71681
            with self.assertRaises(RuntimeError):launch.check_resources(current,baseline,456)
            current=deepcopy(baseline);current[launch.GPU_UUIDS[3]]['processes']=[{'pid':456,'used_mib':1}]
            with self.assertRaises(RuntimeError):launch.check_resources(current,baseline,456)
        current=deepcopy(baseline);current[launch.GPU_UUID]['free_mib']=79000
        with self.assertRaises(RuntimeError):launch.check_resources(current)

    def fixture(self,root):
        folder=root/'calibration';folder.mkdir();png=root/'raw.png';Image.new('RGB',(480,480),(5,10,50)).save(png)
        train=[{'id':str(i),'query':'the radio','label':'PNU'[i%3],'path':str(png),'png_sha256':launch.sha(png)} for i in range(81)]
        repeat=train[:16];data={'fixed':'data'};identity={'code_commit':'fixed','config':launch.CONFIG,'environment':{'fixed':'environment'},
             'model_identity':{'path':'fixed-model'},'dataset':data,'train_ids':[r['id'] for r in train],
             'repeat_ids':[r['id'] for r in repeat],'training_updates':0,'new_controls':0}
        (root/'launch.json').write_text(json.dumps({k:identity[k] for k in ('model_identity','environment')}))
        (folder/'identity.json').write_text(json.dumps(identity));predictions=[];batches=[]
        for phase,group,size in (('primary',train,4),('repeat',repeat,8)):
            for i in range(0,len(group),size):batches.append({'phase':phase,'ids':[r['id'] for r in group[i:i+size]],'batch_size':len(group[i:i+size]),'seconds':1.,'padded_input_tokens':100})
            for row in group:
                _,receipt=messages(row['query'],Image.open(png));state={'P':'present','N':'absent','U':'uncertain'}[row['label']]
                parsed={'visibility':state,'boxes':[[100,200,300,400]] if state=='present' else []}
                decoded=decode_response(SimpleNamespace(tokenizer=FixtureTokenizer()),[ord(c)+10 for c in json.dumps(parsed)]+[1])
                predictions.append({'id':row['id'],'phase':phase,'query':row['query'],'expected_visibility':row['label'],
                    'png_sha256':row['png_sha256'],**receipt,'input_tokens':100,'input_ids_sha256':'a'*64,
                    **decoded,
                    'batch_seconds':1.,'training_eligible':False,'manual_box_review':'PENDING'})
        ledger=folder/'predictions.jsonl';ledger.write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        result={'status':'complete','code_commit':'fixed','config':launch.CONFIG,'generated_examples':97,'primary_unique_images':81,
                'repeated_images':16,'exact_repeat_agreements':16,'visibility':worker.report(predictions[:81],train),'batches':batches,
                'wall_seconds':100.,'cuda_peak_reserved_mib':60000,'predictions_sha256':launch.sha(ledger),'manual_box_review':'PENDING',
                'training_label_release':False,'training_updates':0,'new_controls':0}
        (folder/'result.json').write_text(json.dumps(result))
        return train,repeat,data,predictions,result

    def test_full_result_validation_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch,'ROOT',Path(tmp)):
            root=Path(tmp);train,repeat,data,predictions,result=self.fixture(root);folder=root/'calibration'
            with patch.object(launch,'training_rows',return_value=(train,repeat,data)),patch.object(launch,'load_processor',return_value=SimpleNamespace(tokenizer=FixtureTokenizer())):
                self.assertEqual(launch.validate_result('fixed'),result)
                for key,value in (('id','foreign-validation'),('training_eligible',True),('query','task0 success'),
                                  ('input_ids_sha256','bad'),('has_eos',False),('resized_pixels_sha256','b'*64),
                                  ('parsed',{'visibility':'absent','boxes':[]}),('action','move')):
                    changed=deepcopy(predictions);changed[0][key]=value
                    ledger=folder/'predictions.jsonl';ledger.write_text(''.join(json.dumps(p)+'\n' for p in changed))
                    (folder/'result.json').write_text(json.dumps({**result,'predictions_sha256':launch.sha(ledger)}))
                    with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):launch.validate_result('fixed')
                ledger.write_text(''.join(json.dumps(p)+'\n' for p in predictions))
                for key,value in (('seconds','bogus'),('seconds',float('nan')),('seconds',2.),('batch_size',99),
                                  ('padded_input_tokens',101),('extra','unauthorized')):
                    altered=deepcopy(result);altered['batches'][0][key]=value
                    (folder/'result.json').write_text(json.dumps(altered))
                    with self.subTest(key=key,value=value),self.assertRaises(ValueError):launch.validate_result('fixed')

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
import dual_teacher_calibration as dual
import calibrate_grounding_teacher as worker
import launch_grounding_calibration as launcher


class Tokenizer:
    eos_token_id=1;pad_token_id=0;unk_token_id=2;all_special_ids=[0,1,2]
    def __len__(self):return 1000
    def convert_tokens_to_ids(self,s):return 1
    def convert_ids_to_tokens(self,i):return '<|im_end|>'
    def encode(self,text,add_special_tokens=False):return [10+ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(i-10) if i>=10 else '<special>' for i in ids)


class DualTeacherTests(unittest.TestCase):
    def tearDown(self):launcher.configure_profile('h81');dual.load_rows.cache_clear()

    def test_new_profile_propagates_and_old_budget_restores(self):
        launcher.configure_profile('h83')
        self.assertEqual(launcher.ROOT,worker.ROOT);self.assertEqual(launcher.CONFIG['protocol'],dual.VERSION)
        self.assertEqual(launcher.CONFIG['max_examples'],300)
        self.assertEqual(launcher.command()[3],'2700s');self.assertIn('h83_dual_teacher_v1',launcher.command()[-1])
        with patch.object(dual,'load_rows',return_value=('rows',[],{})):
            self.assertEqual(worker.training_rows(),('rows',[],{}))
        launcher.configure_profile('h82');self.assertEqual(launcher.command()[3],'1800s')
        self.assertEqual(launcher.CONFIG['max_examples'],97)
        self.assertEqual(launcher.CONFIG['max_gpu_mib'],71680)

    def fixtures(self):
        review=json.loads(dual.REVIEW.read_text());first=json.loads(dual.references.RAW_REVIEW.read_text())
        refs=json.loads(dual.references.SPEC.read_text())
        historical=set(map(tuple,review['excluded_groups']))-{(g['task'],g['instance']) for g in first['groups']}
        plan={'source_identity':{'historical_train_groups':sorted(historical)}}
        states={r['id']:{**{k:r[k] for k in ('id','task','instance','episode','frame','split','images')},
                    'image_receipts':{v:{'png_sha256':s} for v,s in r['png_sha256'].items()}} for r in review['rows']}
        return review,plan,states,first,{'references':refs['references']}

    def test_gold_is_pre_model_and_fixed_fresh_sources(self):
        review,plan,states,first,refs=self.fixtures()
        rows=dual.validate_review(review,plan,states,first,refs)
        self.assertEqual(len(rows),150);self.assertEqual(sum(r['stratum']=='core_targets' for r in rows),90)
        self.assertEqual(sum(r['label']=='P' for r in rows),37)
        for key,value in (('training_eligible',True),('teacher_outputs_seen_before_labeling',True),
                          ('heldout_validation_or_test_used',True),('protect_groups_in_future_student_releases',[]),
                          ('excluded_groups',[]),('counts',{'P':150})):
            with self.subTest(key=key),self.assertRaises(ValueError):
                dual.validate_review({**review,key:value},plan,states,first,refs)
        bad=deepcopy(review);bad['rows'][0]['query']='task success'
        with self.assertRaises(ValueError):dual.validate_review(bad,plan,states,first,refs)
        bad=deepcopy(states);bad[review['rows'][0]['id']]['split']='validation'
        with self.assertRaises(ValueError):dual.validate_review(review,plan,bad,first,refs)
        bad=deepcopy(states);bad[review['rows'][0]['id']]['image_receipts']['head']['png_sha256']='changed'
        with self.assertRaises(ValueError):dual.validate_review(review,plan,bad,first,refs)
        with self.assertRaises(ValueError):
            dual.validate_review(review,plan,states,first,{'references':[{'task':rows[0]['task'],'instance':rows[0]['instance']}]})

    def test_presence_decoder_never_hides_truncation_extra_tokens_or_json(self):
        processor=SimpleNamespace(tokenizer=Tokenizer())
        # Fixture is character-level; use short native token aliases to stay in
        # the actual 32-token cap while still checking the exact lookup contract.
        encoded={(4,5,1):'P',(4,6,1):'N',(4,7,1):'U'}
        processor.tokenizer.decode=lambda ids,skip_special_tokens=False:dual.presence.TARGETS[{5:'P',6:'N',7:'U'}[ids[-1]]]
        with patch.object(dual,'presence_tokens',return_value=encoded):
            got=dual.decode_presence(processor,[4,5,1,0,0]);self.assertTrue(got['format_valid'])
            self.assertEqual(got['parsed'],{'visibility':'present'})
            for ids in ([4,5],[4,5,1,9],[4,8,1],[2,4,5,1],[4,5,1]+[0]*31):
                with self.subTest(ids=ids):self.assertFalse(dual.decode_presence(processor,ids)['format_valid'])

    def test_agreement_drops_uncertainty_and_disagreement_not_gold_errors(self):
        rows=[];large=[];small=[]
        def add(i,truth,a,b,stratum='core_targets',valid=True):
            rows.append({'id':str(i),'label':truth,'stratum':stratum})
            large.append({'id':str(i),'format_valid':valid,'parsed':{'visibility':dual.presence.CLASSES[a],'boxes':[[1,2,3,4]] if a=='P' else []}})
            small.append({'id':str(i),'format_valid':True,'parsed':{'visibility':dual.presence.CLASSES[b]}})
        for i in range(20):add(i,'P','P','P')
        for i in range(20,40):add(i,'N','N','N')
        add(40,'P','U','U');add(41,'P','P','N');add(42,'U','N','N')
        add(43,'N','N','N',valid=False)
        for i in range(60):add(i+100,'N','N','N','cross_task_negative_control')
        report=dual.agreement_report(large,small,rows)
        self.assertTrue(report['visibility_candidate_gate_passed'])  # 20/21 N >=95%
        self.assertEqual(report['strata']['core_targets']['errors'],['42'])
        self.assertFalse(report['training_label_release'])
        self.assertFalse(report['proposals'][40]['accepted_candidate'])
        self.assertFalse(report['proposals'][41]['accepted_candidate'])
        self.assertFalse(report['proposals'][43]['accepted_candidate'])
        # Extra perfect cross-task negatives cannot rescue a bad core score.
        rows[20]['label']='U'
        report=dual.agreement_report(large,small,rows)
        self.assertFalse(report['visibility_candidate_gate_passed'])
        self.assertLess(report['strata']['core_targets']['classes']['N']['precision'],.95)
        self.assertTrue(all(p['training_eligible'] is False for p in report['proposals']))

    def test_candidate_boxes_stay_from_corresponding_grounder(self):
        rows=[{'id':'p','label':'P','stratum':'core_targets'},{'id':'n','label':'N','stratum':'cross_task_negative_control'}]
        g=[{'id':r['id'],'format_valid':True,'parsed':{'visibility':dual.presence.CLASSES[r['label']],
              'boxes':[[30,40,50,60]] if r['label']=='P' else []}} for r in rows]
        c=[{'id':r['id'],'format_valid':True,'parsed':{'visibility':dual.presence.CLASSES[r['label']]}} for r in reversed(rows)]
        out=dual.agreement_report(g,c,rows)
        self.assertEqual(out['proposals'][0]['boxes'],[[30,40,50,60]])
        self.assertEqual(out['proposals'][1]['boxes'],[])

    def test_final_validator_reconstructs_both_encodings_tokens_metrics_and_unloads(self):
        processor=SimpleNamespace(tokenizer=Tokenizer());launcher.configure_profile('h83');cfg=launcher.CONFIG
        def encode(_processor,row,_image):
            return None,{'protocol':'fixture-image-only','input_tokens':100+int(row['id'])%3,
                         'input_ids_sha256':dual.hashlib.sha256(row['id'].encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'calibration';folder.mkdir();png=root/'raw.png';Image.new('RGB',(480,480)).save(png)
            rows=[{'id':str(i),'query':'the radio','label':'PNU'[i%3],'stratum':'core_targets' if i<90 else 'cross_task_negative_control',
                   'path':str(png),'png_sha256':dual.sha(png)} for i in range(150)]
            data={'fixture':'fresh-source'};model={'path':'fixed27B'};env={'python':'frozen'}
            launch={'model_identity':model,'environment':env};(root/'launch.json').write_text(json.dumps(launch))
            identity={'code_commit':'fixed','config':cfg,'environment':env,'dataset':data,'model_identity':model,
                      'ids':[r['id'] for r in rows],'training_updates':0,'new_controls':0}
            (folder/'identity.json').write_text(json.dumps(identity));pred=[];batches=[]
            for phase in ('grounding','presence'):
                for start in range(0,150,8):
                    group=rows[start:start+8]
                    batches.append({'phase':phase,'ids':[r['id'] for r in group],'seconds':1.,'batch_size':len(group),
                                    'padded_input_tokens':max(encode(None,r,None)[1]['input_tokens'] for r in group)})
                    for r in group:
                        value={'visibility':dual.presence.CLASSES[r['label']]}
                        if phase=='grounding':value['boxes']=[[100,200,300,400]] if r['label']=='P' else []
                        text=json.dumps(value,separators=(',',':'));tokens=processor.tokenizer.encode(text)+[1]
                        decoded=(dual.decode_response if phase=='grounding' else dual.decode_presence)(processor,tokens)
                        self.assertTrue(decoded['format_valid'])
                        pred.append({'id':r['id'],'phase':phase,'query':r['query'],'expected_visibility':r['label'],
                            'stratum':r['stratum'],'png_sha256':r['png_sha256'],**encode(None,r,None)[1],**decoded,
                            'batch_seconds':1.,'training_eligible':False,'manual_box_review':'PENDING'})
            result={'status':'complete','code_commit':'fixed','config':cfg,'generated_examples':300,'unique_images':150,
                'training_label_release':False,'training_updates':0,'new_controls':0,'wall_seconds':100.,'cuda_peak_reserved_mib':60000.,
                'batches':batches,'grounding':worker.report(pred[:150],rows),'critic':worker.report(pred[150:],rows),
                'agreement':dual.agreement_report(pred[:150],pred[150:],rows),
                'model_unloads':[{'phase':p,'allocated_mib':0.,'reserved_mib':0.} for p in ('grounding','presence')]}
            def write(chosen,summary):
                (folder/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in chosen))
                (folder/'result.json').write_text(json.dumps({**summary,'predictions_sha256':dual.sha(folder/'predictions.jsonl')}))
            write(pred,result)
            with patch.object(dual,'load_rows',return_value=(rows,[],data)),patch.object(dual,'load_processor',return_value=processor),\
                 patch.object(dual.references,'encode',side_effect=encode),patch.object(dual.presence,'encode',side_effect=encode):
                self.assertEqual(dual.validate_result(root,'fixed',cfg)['generated_examples'],300)
                for key,value in (('id','foreign'),('input_ids_sha256','changed'),('query','task status'),('training_eligible',True),
                                  ('generated_token_ids',[2,1]),('phase','grounding'),('reference_inputs',['answer'])):
                    altered=deepcopy(pred);altered[150][key]=value;write(altered,result)
                    with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):
                        dual.validate_result(root,'fixed',cfg)
                for key,value in (('model_unloads',[]),('generated_examples',299),('agreement',{})):
                    write(pred,{**result,key:value})
                    with self.subTest(key=key),self.assertRaises(ValueError):dual.validate_result(root,'fixed',cfg)

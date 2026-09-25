from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/vlm_sft'),str(REPO/'src')]
import visual_presence as visual
import launch_visual_presence as launch


class VisualPresenceTests(unittest.TestCase):
    def test_tensor_prefix_response_eos_and_tail_loss_match_native_shift(self):
        import torch
        from modeling import collate,supervised_loss
        class Tokenizer:
            eos_token_id=1;unk_token_id=0;pad_token_id=0
            def convert_tokens_to_ids(self,text):return 1 if text=='<|im_end|>' else 0
            def convert_ids_to_tokens(self,idx):return '<|im_end|>' if idx==1 else 'other'
            def encode(self,text,**_):return [ord(c)+10 for c in text]
        class Processor:
            tokenizer=Tokenizer()
            def apply_chat_template(self,msg,**kwargs):
                self.last=msg
                assert kwargs['enable_thinking'] is False
                n=3 if msg[1]['content'][1]['image'].width==640 else 5
                return {'input_ids':torch.tensor([[4]*n]),'attention_mask':torch.ones((1,n),dtype=torch.long),
                        'mm_token_type_ids':torch.ones((1,n),dtype=torch.long),'pixel_values':torch.ones((2,3)),
                        'image_grid_thw':torch.tensor([[1,2,2]])}
        processor=Processor();items=[]
        for label,size in (('P',720),('U',480)):
            row={'query':'the radio','label':label,'target':visual.TARGETS[label],
                 'instance':999,'frame':12345,'history':['RIGHT_UP']*9}
            im=Image.new('RGB',(size,size))
            prefix,_=visual.encode(processor,row,im)
            full,_=visual.encode(processor,row,im,supervised=True);n=prefix['input_ids'].shape[1]
            self.assertTrue(torch.equal(full['input_ids'][:,:n],prefix['input_ids']))
            self.assertTrue((full['labels'][:,:n]==-100).all());self.assertEqual(int(full['labels'][0,-1]),1)
            self.assertTrue((full['mm_token_type_ids'][:,n:]==0).all());items.append(full)
            self.assertNotIn('12345',str(processor.last));self.assertNotIn('RIGHT_UP',str(processor.last))
            with self.assertRaises(ValueError):visual.encode(processor,row,im,supervised=True,blind=True)
        batch=collate(items,0)
        self.assertTrue((batch['labels'][batch['attention_mask']==0]==-100).all())
        class Model:
            def __call__(self,input_ids,labels=None,logits_to_keep=None,**_):
                shape=(*input_ids.shape,160)
                logits=torch.sin(torch.arange(input_ids.numel()*160).reshape(shape).float()/100)
                loss=None
                if labels is not None:loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,160),labels[:,1:].reshape(-1),ignore_index=-100)
                return SimpleNamespace(logits=logits[:,-logits_to_keep:] if logits_to_keep else logits,loss=loss)
        model=Model();self.assertTrue(torch.allclose(model(**batch).loss,supervised_loss(model,batch),atol=1e-6))

    def test_prefix_has_one_raw_view_and_query_not_metadata(self):
        image=Image.new('RGB',(720,720),(4,5,6))
        msg,receipt=visual.messages('the radio',image)
        self.assertEqual(receipt['size'],[640,640]);self.assertEqual(image.size,(720,720))
        content=msg[1]['content'];self.assertEqual([x['type'] for x in content],['text','image','text'])
        self.assertEqual(content[0]['text'],'CURRENT_RAW');self.assertEqual(content[-1]['text'],'Query: the radio')
        _,wrist=visual.messages('the radio',Image.new('RGB',(480,480)))
        self.assertEqual(wrist['size'],[480,480])
        _,blind=visual.messages('the radio',image,blind=True)
        self.assertNotEqual(receipt['resized_pixels_sha256'],blind['resized_pixels_sha256'])
        self.assertEqual(receipt['size'],blind['size'])
        with self.assertRaises(ValueError):visual.messages('hidden_object_123',image)
        with self.assertRaises(ValueError):visual.messages('the radio',Image.new('RGB',(256,256)))

    def test_real_parent_coverage_hash_and_source_split(self):
        self.assertEqual(visual.sha(visual.REVIEW),visual.REVIEW_SHA)
        review=json.loads(visual.REVIEW.read_text())
        self.assertEqual(len(review['rows']),36)
        self.assertEqual(len({r['id'] for r in review['rows']}),36)
        counts={k:sum(r['labels'].count(k) for r in review['rows']) for k in visual.CLASSES}
        self.assertEqual(counts,{'P':43,'N':54,'U':11})
        self.assertEqual(review['approved_for'],'bounded_visual_presence_pilot_only')
        self.assertIn('task_completion',review['not_approved_for'])

    def test_metrics_count_unknown_and_absence_separately(self):
        rows=[{'id':'a','label':'P'},{'id':'b','label':'N'},{'id':'c','label':'U'}]
        result=visual.metrics(rows,{'a':'P','b':'N','c':'N'})
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertEqual(result['recall'],{'P':1,'N':1,'U':0})
        with self.assertRaises(ValueError):visual.metrics(rows,{'a':'P','b':'N'})
        with self.assertRaises(ValueError):visual.metrics(rows,{'a':'P','b':'N','c':'DONE'})
        self.assertFalse(result['full_task_success_rate_claim'])

    def resources(self):
        return {u:{'used_mib':0,'free_mib':81152,'processes':[]} for u in launch.GPU_UUIDS}

    def test_exclusive_gpu2_ownership_and_preserved_teammates(self):
        base=self.resources();base[launch.GPU_UUIDS[0]]['processes']=[{'pid':123,'used_mib':12548}]
        launch.check_resources(base)
        current=deepcopy(base);current[launch.GPU_UUID]['processes']=[{'pid':456,'used_mib':16000}]
        with patch.object(launch,'belongs_to_session',side_effect=lambda pid,sid:pid==sid):
            launch.check_resources(current,base,456)
            with self.assertRaises(RuntimeError):launch.check_resources(current,base,456,released=True)
            with self.assertRaises(RuntimeError):launch.check_resources(current,base,789)
            current[launch.GPU_UUID]['processes'][0]['used_mib']=24577
            with self.assertRaises(RuntimeError):launch.check_resources(current,base,456)
            current=deepcopy(base);current[launch.GPU_UUIDS[3]]['processes']=[{'pid':456,'used_mib':1}]
            with self.assertRaises(RuntimeError):launch.check_resources(current,base,456)
        occupied=deepcopy(base);occupied[launch.GPU_UUID]['processes']=[{'pid':123,'used_mib':1}]
        with self.assertRaises(RuntimeError):launch.check_resources(occupied)

    def test_foreign_evaluation_or_79_updates_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch,'ROOT',Path(tmp)):
            folder=Path(tmp)/'training';folder.mkdir()
            value={'status':'complete','code_commit':'code','optimizer_updates':79,'generated_calls':110,
                   'config':launch.CONFIG,'full_task_success_rate_claim':False,'online_actor_evaluated':False,'wall_seconds':100}
            with self.assertRaises(ValueError):launch.validate_result(value,'code')
            value['optimizer_updates']=80;value['online_actor_evaluated']=True
            with self.assertRaises(ValueError):launch.validate_result(value,'code')

    def test_fixed_training_is_not_loading_an_old_adapter_or_resuming(self):
        from train_visual_presence import CONFIG
        self.assertEqual(CONFIG['updates'],80);self.assertEqual(CONFIG['lora_rank'],8)
        self.assertEqual(CONFIG['max_generated_calls'],112)
        self.assertEqual(CONFIG['wall_seconds'],1800)
        self.assertEqual(CONFIG['effective_batch']//CONFIG['microbatch'],4)
        self.assertNotIn('adapter',CONFIG)

    def complete_fixture(self,root):
        folder=root/'training';folder.mkdir();path=root/'raw.png';Image.new('RGB',(480,480),(10,20,30)).save(path)
        rows=[{'id':str(i),'split':'visual_train' if i<81 else 'visual_validation','query':'the radio',
               'view':'left_wrist','path':str(path),'png_sha256':visual.sha(path),'label':'PNU'[i%3]} for i in range(108)]
        train,valid=rows[:81],rows[81:];data={'exact':'data'};base={'exact':'model'}
        def write(name,value): (root/name).write_text(json.dumps(value))
        def receipt(row,blind=False):
            _,r=visual.messages(row['query'],visual.checked_image(row),blind=blind)
            return {'id':row['id'],**r,'input_ids_sha256':'a'*64,'input_tokens':100,'supervised_tokens':20,
                    'prediction':row['label'],'text':visual.TARGETS[row['label']],'expected':row['label'],'latency_seconds':.1,'output_tokens':10}
        write('launch.json',{'dataset':data,'model_identity':base,'environment':{'frozen':'environment'}})
        write('training/identity.json',{'code_commit':'code','config':launch.CONFIG,'data':data,'base_model_identity':base,
              'environment':{'frozen':'environment'},'old_adapter_loaded':False,'train_counts':{'P':27,'N':27,'U':27},'physical_gpu':2,'images_per_input':1,'action_or_success_labels':False})
        write('training/mask_gate.json',[receipt(r) for r in train])
        write('training/loss_gate.json',{'native':1.,'custom':1.,'absolute_error':0.,'ids':['0','1'],'all_labels_reviewed':True,'left_padding':True})
        sampler=random.Random(41)
        steps=[{'step':i,'sample_ids':[r['id'] for _ in range(4) for r in sampler.sample(train,2)],'loss':1.,'gradient_norm':1.,'elapsed_seconds':i} for i in range(1,81)]
        (folder/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in steps))
        write('training/restore_gate.json',{'passed':True,'changed_lora_tensors':2,'max_logit_error':0.,'before':receipt(train[0]),'restored':receipt(train[0])})
        result={'status':'complete','code_commit':'code','optimizer_updates':80,'generated_calls':110,'config':launch.CONFIG,
                'full_task_success_rate_claim':False,'online_actor_evaluated':False,'wall_seconds':100,'cuda_peak_reserved_mib':1000,
                'dataset':data,'base_model_identity':base,'environment':{'frozen':'environment'},'checkpoints':[],**visual.train_prior_metrics(train,valid)}
        for step in (2,80):
            adapter=folder/f'adapter_{step:04d}';adapter.mkdir()
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture');(adapter/'adapter_config.json').write_text('{}')
            result['checkpoints'].append({'step':step,'path':str(adapter),'weight_sha256':visual.sha(adapter/'adapter_model.safetensors'),
                                         'config_sha256':visual.sha(adapter/'adapter_config.json')})
        for name,key,step,blind in (('base_images','base',0,False),('base_gray','base_gray',0,True),
                                  ('finetuned_images','finetuned',80,False),('finetuned_gray','finetuned_gray',80,True)):
            predictions=[receipt(r,blind) for r in valid];score=visual.metrics(valid,{r['id']:r['prediction'] for r in predictions})
            write('training/'+name+'.json',{'step':step,'blind':blind,'predictions':predictions,'metrics':score});result[key]=score
        return rows,data,result

    def test_full_acceptance_requires_training_numerical_and_pixel_evidence(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(launch,'ROOT',Path(temp)):
            root=Path(temp);rows,data,result=self.complete_fixture(root)
            with patch.object(launch,'load_rows',return_value=(rows,data)):
                launch.validate_result(result,'code')
                mutations=[('mask_gate.json',lambda d:d.pop()),
                           ('loss_gate.json',lambda d:d.update(custom=2.)),
                           ('restore_gate.json',lambda d:d.update(changed_lora_tensors=0)),
                           ('identity.json',lambda d:d.update(old_adapter_loaded=True)),
                           ('base_gray.json',lambda d:d['predictions'][0].update(blind=False)),
                           ('base_images.json',lambda d:d['predictions'][0].update(resized_pixels_sha256='b'*64)),
                           ('finetuned_images.json',lambda d:d['predictions'][0].update(input_ids_sha256='b'*64))]
                for name,change in mutations:
                    path=root/'training'/name;raw=path.read_text();obj=json.loads(raw);change(obj);path.write_text(json.dumps(obj))
                    with self.subTest(name=name),self.assertRaises(ValueError):launch.validate_result(result,'code')
                    path.write_text(raw)
                for wall in (-1,True,float('nan'),1801):
                    with self.subTest(wall=wall),self.assertRaises(ValueError):launch.validate_result({**result,'wall_seconds':wall},'code')
                broken=deepcopy(result);broken['query_size_majority_baseline']['accuracy']=.123
                with self.assertRaises(ValueError):launch.validate_result(broken,'code')
                (root/'training/loss_gate.json').rename(root/'training/loss_gate.missing')
                with self.assertRaises(FileNotFoundError):launch.validate_result(result,'code')


if __name__=='__main__':unittest.main()

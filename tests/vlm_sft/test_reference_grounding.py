from copy import deepcopy
import hashlib
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
import reference_grounding as ref
import calibrate_grounding_teacher as worker
import launch_grounding_calibration as launch


class ReferenceGroundingTests(unittest.TestCase):
    def tearDown(self):launch.configure_profile('h81');ref.gallery.cache_clear()

    def test_profile_is_explicit_and_propagates_to_worker_command(self):
        old=launch.command();launch.configure_profile('h82')
        self.assertEqual(launch.ROOT,worker.ROOT)
        self.assertEqual(launch.CONFIG,worker.CONFIG)
        self.assertEqual(launch.CONFIG['protocol'],ref.VERSION)
        self.assertEqual(launch.command()[launch.command().index('--profile')+1],'h82')
        self.assertIn('h82_reference_calibration_v1',launch.command()[-1])
        self.assertEqual(launch.command()[3],'1800s')
        launch.configure_profile('h81');self.assertEqual(launch.command(),old)
        with self.assertRaises(ValueError):launch.configure_profile('foreign')

    def test_parent_review_exact_train_references_and_reject_tampering(self):
        spec=json.loads(ref.SPEC.read_text());review=json.loads(ref.RAW_REVIEW.read_text())
        self.assertEqual(len(ref.validate_spec(spec,review)),7)
        for key,value in (('split','validation'),('id','foreign'),('png_sha256','x'),('path','../other.png'),('query','task progress')):
            altered=deepcopy(spec);altered['references'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ref.validate_spec(altered,review)
        spec['approved_for']='PENDING_CROP_REVIEW'
        with self.assertRaises(ValueError):ref.validate_spec(spec,review)

    def test_three_references_then_exact_current_and_no_answer_leak(self):
        spec=json.loads(ref.SPEC.read_text());refs=spec['references']
        images={r['name']:Image.new('RGB',tuple(r['reviewed_size']),(n,20,30)) for n,r in enumerate(refs)}
        identity={'spec_sha256':'fixed','references':[{'name':r['name'],'id':r['id']} for r in refs]}
        raw=Image.new('RGB',(720,720),(3,4,5))
        with patch.object(ref,'gallery',return_value=(refs,images,identity)):
            for query in ref.QUERIES:
                msg,receipt=ref.messages(query,raw);content=msg[1]['content']
                imgs=[c['image'] for c in content if c['type']=='image']
                self.assertEqual(len(imgs),4);self.assertEqual(imgs[-1].size,(640,640))
                self.assertEqual(receipt['resized_pixels_sha256'],hashlib.sha256(imgs[-1].tobytes()).hexdigest())
                text=' '.join(c['text'] for c in content if c['type']=='text')
                self.assertTrue(text.index('NEGATIVE_ROBOT_REFERENCE')<text.index('CURRENT_RAW'))
                self.assertFalse(any(r['id'] in text for r in refs))
                self.assertNotIn('expected_visibility',text)
                self.assertEqual(len(receipt['reference_inputs']),3)
            with self.assertRaises(ValueError):ref.messages('the radio',raw,blind=True)
            with self.assertRaises(ValueError):ref.messages('a candle',raw)

    def test_crops_checked_no_upscale_and_bad_crop_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'raw.png';Image.new('RGB',(480,480),(1,2,3)).save(path)
            r={'path':'raw.png','png_sha256':ref.sha(path),'original_size':[480,480],'crop_xyxy':[4,6,104,66]}
            self.assertEqual(ref.crop_reference(root,r).size,(100,60))
            for crop in ([0,0,481,5],[0,0,5,5.0],[6,4,3,8]):
                with self.assertRaises(ValueError):ref.crop_reference(root,{**r,'crop_xyxy':crop})

    def test_reference_source_group_not_calibration(self):
        rows=[{'id':str(i),'task':0,'instance':i//9,'split':'visual_train'} for i in range(81)]
        launch.configure_profile('h82')
        with patch.object(worker,'load_rows',return_value=(rows,{'dataset':'frozen'})):
            with patch.object(ref,'identity',return_value={'references':[{'task':0,'instance':99}]}):
                self.assertIn('teacher_references',worker.training_rows()[2])
            with patch.object(ref,'identity',return_value={'references':[{'task':0,'instance':0}]}):
                with self.assertRaises(ValueError):worker.training_rows()

    def test_encoder_uses_four_images_and_enforces_context(self):
        import torch
        processor=SimpleNamespace(apply_chat_template=lambda *a,**kw:{
            'input_ids':torch.ones((1,100),dtype=torch.long),'image_grid_thw':torch.ones((4,3))})
        with patch.object(ref,'messages',return_value=([],{})):
            _,receipt=ref.encode(processor,{'query':'the radio'},None)
            self.assertEqual(receipt['input_tokens'],100)
            for n,grids in ((1801,4),(100,1)):
                processor.apply_chat_template=lambda *a,**kw:{'input_ids':torch.ones((1,n),dtype=torch.long),'image_grid_thw':torch.ones((grids,3))}
                with self.assertRaises(ValueError):ref.encode(processor,{'query':'the radio'},None)

    def test_source_split_frame_and_image_not_self_reported(self):
        r={'id':'t0_i4_e5_f000060_head','task':0,'instance':4,'path':'images/t0_i4_e5_f000060_head.png',
           'png_sha256':'fixed','original_size':[720,720]}
        source={'task':0,'instance':4,'episode':5,'split':'train','selected_frames':[60]}
        state={'task':0,'instance':4,'episode':5,'frame':60,'split':'train','images':{'head':r['path']},
               'image_receipts':{'head':{'png_sha256':'fixed','resolution':[720,720]}}}
        states={'t0_i4_e5_f000060':state};plan={'sources':[source]}
        ref.validate_sources([r],plan,states)
        for key,value in (('split','validation'),('selected_frames',[61]),('episode',99)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                ref.validate_sources([r],{'sources':[{**source,key:value}]},states)
        with self.assertRaises(ValueError):
            ref.validate_sources([r],plan,{'t0_i4_e5_f000060':{**state,'split':'test'}})

import json
from pathlib import Path
import sys
import types
import unittest
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).resolve().parents[4]/'src')]
from fixed_skill_protocol import FixedSkillSession,low_prefix,validate_rgb_proprio


class FixedTests(unittest.TestCase):
    def setUp(self):
        self.bundle=json.dumps([dict(verb='GRASP',target='radio',source='',destination='',target_part='',
                                    arm='RIGHT',unbound_relation='')],sort_keys=True,separators=(',',':'))
        self.cases={'radio':dict(manifest_sha256='a'*64,seeds=[17],context_id='ctx',start_control=32,end_control=160,
                                semantic_bundle=self.bundle)}
        self.models={'parent':{'sha256':'b'*64},'candidate':{'sha256':'c'*64}}
        self.begin=dict(op='begin',case='radio',model='parent',episode='run1',seed=17,manifest_sha256='a'*64)
        self.request=dict(op='action',episode='run1',control_step=32,context_id='ctx',images={},proprio=None)

    def test_session_clock_context_and_reset_isolation(self):
        s=FixedSkillSession(self.cases,self.models)
        with self.assertRaises(ValueError):s.check_action(self.request)
        self.assertEqual(s.begin(self.begin)['model_sha256'],'b'*64)
        with self.assertRaises(ValueError):s.begin(self.begin)
        for key,bad in [('context_id','wrong'),('episode','other'),('control_step',48),('reward',1)]:
            with self.assertRaises(ValueError):s.check_action(dict(self.request,**{key:bad}))
        s.check_action(self.request);s.acknowledge_chunk()
        with self.assertRaises(ValueError):s.check_action(self.request)
        s.check_action(dict(self.request,control_step=48))
        other=FixedSkillSession(self.cases,self.models);other.begin(dict(self.begin,episode='run2',model='candidate'))
        self.assertEqual(other.next_control,32);self.assertEqual(s.next_control,48)

    def test_observation_whitelist(self):
        images={k:np.zeros((3,224,224),dtype=np.uint8) for k in ['head_rgb','left_wrist_rgb','right_wrist_rgb']}
        validate_rgb_proprio(images,np.zeros(61,dtype=np.float32))
        for bad in (np.zeros(60),np.full(61,np.nan)):
            with self.assertRaises(ValueError):validate_rgb_proprio(images,bad)
        with self.assertRaises(ValueError):validate_rgb_proprio(dict(images,depth=np.zeros((3,224,224))),np.zeros(61))

    def test_target_free_low_matches_training_prefix(self):
        builder=types.SimpleNamespace(num_input_images=3,_image_sizes={k:(224,224) for k in
            ['head_rgb','left_wrist_rgb','right_wrist_rgb']},template='<parent_goal_text_!><active_skills_text_text_!><EOC><eos>',
            hardcode_instruction=None,hardcode_proprio_pad_zeros=False,embodiment_type='galaxea_r1pro')
        prepared=dict(_instructions='turning on radio',proprio='observed',proprio_dim_is_pad='mask',
                      action='teacher must not leak',physical_audit='oracle must not leak')
        out=low_prefix(builder,prepared,task='turning on radio',parent_goal='Task goal: turning on radio',semantic_bundle=self.bundle)
        self.assertEqual(out['template'],builder.template.split('<EOC>')[0]+'<EOC>')
        self.assertNotIn('action',out);self.assertNotIn('physical_audit',out);self.assertFalse(out['low_action_supervision_mask'])
        with self.assertRaises(ValueError):low_prefix(builder,prepared,task='wrong',parent_goal='Task goal: turning on radio',semantic_bundle=self.bundle)


if __name__=='__main__':unittest.main()

from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_short_start_feature_pair import compare


class ShortStartPairTests(unittest.TestCase):
    def fixture(self):
        now=dict(sample_id='now',served_controls=8,control_step=40,task_name='t',
            parent_goal='goal',issued_bundle='same',member_index=0)
        start=dict(now,sample_id='start',served_controls=0,control_step=32)
        old=dict(history_protocol='cadence16_v1',initial_head={'a':torch.ones(2)},
            requests=[dict(request_id='r',sample_id='now',checks=[now])],
            features={'r':dict(context=torch.ones(1,2),proprio=torch.ones(1,2),steps=torch.tensor([40]))})
        new=deepcopy(old);new['history_protocol']='attempt_start_short_v1'
        new['requests'][0]['checks']=[start,now]
        new['features']['r']=dict(context=torch.tensor([[0.,0.],[1.,1.]]),
            proprio=torch.tensor([[0.,0.],[1.,1.]]),steps=torch.tensor([32,40]))
        return old,new

    def test_only_adds_true_attempt_start_and_keeps_all_current_features(self):
        a,b=self.fixture();r=compare(a,b)
        self.assertEqual((r['unchanged'],r['extended_same_attempt_starts']),(0,1))
        c=deepcopy(a);c['history_protocol']='attempt_start_short_v1'
        self.assertEqual(compare(a,c)['unchanged'],1)

    def test_changed_current_wrong_start_repeated_frame_or_new_label_is_rejected(self):
        a,b=self.fixture()
        for mutation in (lambda x:x['features']['r']['context'].fill_(2),
            lambda x:x['requests'][0]['checks'][0].update(served_controls=1),
            lambda x:x['requests'][0]['checks'][0].update(sample_id='now'),
            lambda x:x['requests'][0]['checks'][0].update(issued_bundle='another'),
            lambda x:x['requests'][0].update(sample_id='wrong-label'),
            lambda x:x['features']['r']['steps'].fill_(40),
            lambda x:x['initial_head']['a'].fill_(2)):
            c=deepcopy(b);mutation(c)
            with self.assertRaises(ValueError):compare(a,c)


if __name__=='__main__':unittest.main()

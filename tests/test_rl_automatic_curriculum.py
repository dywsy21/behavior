from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.curriculum import check_curriculum


def fixture():
    spec=dict(worker=0,gpu=2,task_id=0,task='turning_on_radio',seed=0,split='train',
        evaluation_only=False,instance=1,prefix_controls=1076,episode=0,
        first_recorded_terminal=1364,actions='/train/actions.npy')
    image=np.zeros((3,64,64),dtype=np.uint8);image[:,2:4,:]=100
    raw=dict(images={name:image.copy() for name in ('head_rgb','left_wrist_rgb','right_wrist_rgb')},
             state={'left_arm':np.zeros(7)},task='turn on the device',embodiment_type='R1Pro',frequency=30)
    row=dict(episode=0,episode_controls=1076,success=False,terminal=False,observation=raw,
        clock={'simulation_time':35.866666,'physics_index':4304},
        reward_state=dict(goal_success=False,potential=.1,geometry_targets=1))
    return spec,row


class AutomaticCurriculumTests(unittest.TestCase):
    def test_accepts_without_any_human_artifact(self):
        spec,row=fixture();receipt=check_curriculum(spec,row,1076)
        self.assertTrue(receipt['accepted'])
        self.assertFalse(receipt['human_review_required'])
        self.assertFalse(receipt['human_signature_created'])

    def test_rejects_terminal_mismatched_prefix_and_eval_leak(self):
        spec,row=fixture()
        for patch in ({'success':True},{'terminal':True},{'episode_controls':1075}):
            with self.assertRaises(ValueError):check_curriculum(spec,dict(row,**patch),1076)
        with self.assertRaises(ValueError):check_curriculum(dict(spec,split='public_test'),row,1076)
        row['observation']['reward_state']={'potential':1}
        with self.assertRaises(ValueError):check_curriculum(spec,row,1076)

    def test_rejects_blank_camera_missing_reward_and_nan_state(self):
        spec,row=fixture()
        bad=deepcopy(row);bad['observation']['images']['head_rgb'][:]=0
        with self.assertRaises(ValueError):check_curriculum(spec,bad,1076)
        bad=deepcopy(row);del bad['reward_state']
        with self.assertRaises(ValueError):check_curriculum(spec,bad,1076)
        bad=deepcopy(row);bad['observation']['state']['left_arm'][0]=np.nan
        with self.assertRaises(ValueError):check_curriculum(spec,bad,1076)


if __name__ == '__main__':unittest.main()

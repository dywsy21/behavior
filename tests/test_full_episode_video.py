import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
from render_full_episode_video import intervals,wrap,timestamp


def fixture():
    chunks=[];controls=[]
    for start in (0,16):
        skill=dict(verb='GRASP',arm='RIGHT',target='radio',destination='')
        display=dict(control_step=start,attempt_generation=1,decision='EXECUTE',
            goal=dict(semantic_bundle=json.dumps([skill])),observer=dict(control_step=start,
                attempt_generation=1,members=[dict(member=0,raw_logits=[0.,1.,2.,3.])]))
        chunks.append(dict(token=str(start),display=display,actions=[[0.]*23]*16))
    for i in range(19):
        controls.append(dict(control_step=i,simulator_apply_ack=True,token=str(i//16*16),
            action_executed_raw23=[0.]*23,terminated=i==18,truncated=False))
    return chunks,controls


class FullVideoTests(unittest.TestCase):
    def test_initial_and_odd_final_frame_and_no_future_boundary_caption(self):
        c,a=fixture();r=intervals(c,a,expected_steps=19)
        self.assertEqual([(v['begin'],v['end']) for v in r],[(0,17),(17,20)])
        self.assertEqual(sum(v['end']-v['begin'] for v in r),20)

    def test_missing_ack_wrong_action_false_terminal_and_foreign_head_fail(self):
        c,a=fixture()
        with self.assertRaises(ValueError):intervals(c,a[:-1],expected_steps=19)
        for key,value in [('simulator_apply_ack',False),('action_executed_raw23',[1.]*23),('terminated',True)]:
            b=copy.deepcopy(a);b[0][key]=value
            with self.assertRaises(ValueError):intervals(c,b,expected_steps=19)
        d=copy.deepcopy(c);d[1]['display']['observer']['attempt_generation']=2
        with self.assertRaises(ValueError):intervals(d,a,expected_steps=19)

    def test_timestamp_and_nonclipping_wrap(self):
        self.assertEqual(timestamp(30),'0:00:01.00')
        self.assertEqual(timestamp(17),'0:00:00.56')
        text='当前意图：'+'右手抓取radio。'*20
        self.assertEqual(''.join(wrap(text)),text)
        self.assertGreater(len(wrap(text)),1)


if __name__=='__main__':unittest.main()

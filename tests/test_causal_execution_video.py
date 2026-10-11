import copy
import importlib.util
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('causal_video',
    Path(__file__).resolve().parents[1]/'scripts/eval/render_causal_execution_video.py')
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)


def fixture():
    job = dict(id='j',case='task_1',seed=17)
    entries = []
    for step, revision in ((0,1),(16,2)):
        goal = dict(semantic_bundle='[{"verb":"GRASP"}]')
        entries.append(dict(kind='plan',job=job,result=dict(reused=False,goal=goal,revision=revision)))
        entries.append(dict(kind='observer',job=job,control_step=step,attempt_generation=revision,
            estimated_feedback='{"estimated_bundle_outcome":"UNKNOWN"}',
            result=dict(members=[dict(raw_logits=[0.,0.,0.,0.],uncalibrated_prediction=
                dict(outcome='IN_PROGRESS',confidence=.25))])))
        entries.append(dict(kind='action',job=job,goal=goal,revision=revision,control_step=step,max_controls=16))
        entries.append(dict(kind='ack',job=job,control_step=step+16))
    entries.append(dict(kind='close',job=job,control_step=32))
    controls = [dict(control_step=t,simulator_apply_ack=True) for t in range(1,33)]
    return entries,controls


class CausalVideoTests(unittest.TestCase):
    def test_end_of_chunk_does_not_show_next_plan_or_head_output_early(self):
        entries,controls=fixture()
        states=video.frame_states(entries,job_id='j',start=0,controls=controls,nframes=16)
        self.assertEqual(states[7]['control_step'],16)
        self.assertEqual(states[7]['revision'],1)
        self.assertEqual(states[7]['observer']['control_step'],0)
        self.assertEqual(states[8]['control_step'],18)
        self.assertEqual(states[8]['revision'],2)
        self.assertEqual(states[8]['observer']['control_step'],16)

    def test_missing_ack_wrong_frame_count_and_unclosed_episode_rejected(self):
        entries,controls=fixture()
        with self.assertRaises(ValueError):
            video.frame_states(entries,job_id='j',start=0,controls=controls,nframes=15)
        bad=copy.deepcopy(controls);bad[3]['simulator_apply_ack']=False
        with self.assertRaises(ValueError):
            video.frame_states(entries,job_id='j',start=0,controls=bad,nframes=16)
        with self.assertRaises(ValueError):
            video.frame_states(entries[:-1],job_id='j',start=0,controls=controls,nframes=16)

    def test_foreign_episode_state_and_future_observer_rejected(self):
        entries,controls=fixture();bad=copy.deepcopy(entries)
        bad[1]['job']=dict(id='j',case='other_2',seed=17)
        with self.assertRaises(ValueError):video.frame_states(bad,job_id='j',start=0,controls=controls,nframes=16)
        bad=copy.deepcopy(entries);bad[1]['control_step']=2
        with self.assertRaises(ValueError):video.frame_states(bad,job_id='j',start=0,controls=controls,nframes=16)

    def test_logits_timestamps_and_caption_control_characters(self):
        self.assertAlmostEqual(sum(video.probabilities([900,901,902,903],1.774)),1.)
        with self.assertRaises(ValueError):video.probabilities([0,math.nan,0,0])
        for frame in range(1000):
            h,m,s=video.stamp(frame).split(':');when=int(h)*3600+int(m)*60+float(s)
            self.assertLessEqual(when,frame/15+1e-10)
            self.assertGreater(when,(frame-1)/15)
        self.assertEqual(video.clean('{\\pos(1,1)}\nother'), '(/pos(1,1)) other')


if __name__=='__main__':unittest.main()

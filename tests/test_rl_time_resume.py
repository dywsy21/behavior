import importlib.util
import json
from math import isinf
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.time_limits import NO_TRAINING_WALL,active_wall_limit,training_deadline
from g05.rl.dense_resume import resume_state,safe_restart_boundary


class TimeResumeTests(unittest.TestCase):
    def test_finite_default_unchanged(self):
        m=dict(max_active_wall_seconds=43200,max_training_seconds=28800,final_eval_reserved_seconds=10800)
        self.assertEqual(active_wall_limit(m),43200)
        self.assertEqual(training_deadline(m,started=100,training_started=700),29500)
        self.assertEqual(training_deadline(m,started=100,training_started=10000),32500)

    def test_none_is_explicit_and_really_unlimited_not_a_large_number(self):
        m=dict(entry='method_dense',time_limit_override=NO_TRAINING_WALL,
               max_active_wall_seconds=None,max_training_seconds=None)
        self.assertIsNone(active_wall_limit(m))
        self.assertTrue(isinf(training_deadline(m,started=0,training_started=100)))
        for key,value in [('entry','method'),('time_limit_override',None),('max_training_seconds',28800)]:
            changed=dict(m,**{key:value})
            with self.assertRaises(ValueError): active_wall_limit(changed)
        for value in (-1,0,float('nan'),float('inf')):
            with self.assertRaises(ValueError): active_wall_limit(dict(max_active_wall_seconds=value))

    def test_actual_supervisor_wait_has_no_timeout(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts/rl'))
        spec=importlib.util.spec_from_file_location('wall_launch_test',root/'scripts/rl/launch.py')
        launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
        with TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'manifest.json').write_text(json.dumps(dict(entry='method_dense',
                time_limit_override=NO_TRAINING_WALL,max_active_wall_seconds=None,max_training_seconds=None)))
            process=SimpleNamespace(pid=123,poll=lambda:0)
            with patch.object(launch,'OUT',out),patch.object(launch,'commit',return_value='test'),\
                    patch.object(launch.subprocess,'check_output',return_value='0, 0\n2, 0\n3, 0\n'),\
                    patch.object(launch.subprocess,'Popen',return_value=process),\
                    patch.object(launch,'model_env',return_value={}),\
                    patch.object(launch.os,'killpg') as kill,\
                    patch.object(process,'wait',create=True,return_value=0) as wait:
                launch.supervise('method_dense')
            wait.assert_called_once_with(timeout=None);kill.assert_not_called()
            self.assertEqual(json.loads((out/'supervisor.json').read_text())['status'],'completed')

    def fixture(self):
        m=dict(max_batches=64,max_training_controls=80000,
               workers=[dict(worker=w,instance=i,prefix_controls=p,first_recorded_terminal=t)
                        for w,i,p,t in [(0,1,1076,1364),(1,138,1096,1192)]])
        rows=[]
        for b in range(2):
            rows.append(dict(batch=b,new_actor_updates=4,checkpoint=f'/run/ckpt{b}.pt',
                controls=4000*(b+1),official_reward=1,nonzero_shaping_controls=2048,
                episodes=[dict(worker=w,instance=i,prefix=p,next_prefix=p,success=(w==0))
                          for w,i,p in [(0,1,1076),(1,138,1096)]]))
        receipt=dict(path='/run/ckpt1.pt',controls=8000,actor_updates=102,critic_updates=24)
        return m,rows,receipt

    def test_cumulative_controls_updates_and_curriculum_streaks_survive(self):
        m,rows,receipt=self.fixture()
        state=resume_state(m,rows,receipt,8032)  # interrupted next-prefix controls still count
        self.assertEqual((state['controls'],state['actor_updates'],state['batches']),(8032,102,2))
        self.assertEqual(state['recent'],{'0':[True,True],'1':[False,False]})
        self.assertEqual(state['successes'],2)

    def test_no_missing_actor_steps_reordered_batches_or_regressed_ledger(self):
        m,rows,receipt=self.fixture()
        for bad in [dict(receipt,actor_updates=101),dict(receipt,critic_updates=20),dict(receipt,controls=7999)]:
            with self.assertRaises(ValueError): resume_state(m,rows,bad,8032)
        with self.assertRaises(ValueError): resume_state(m,rows[::-1],receipt,8032)
        with self.assertRaises(ValueError): resume_state(m,rows,receipt,7999)

    def test_stop_only_in_next_prefix_after_matching_durable_checkpoint(self):
        _,rows,receipt=self.fixture(); batch=rows[-1]
        status=dict(phase='training_curriculum',batch=2,batches=2,actor_updates=102,critic_updates=24)
        self.assertTrue(safe_restart_boundary(status,batch,receipt))
        replay=dict(status,replay_progress={'0':16,'1':16});del replay['batch']
        self.assertTrue(safe_restart_boundary(replay,batch,receipt))
        for key,value in [('phase','updating'),('phase','collecting'),('actor_updates',103),('batches',1)]:
            self.assertFalse(safe_restart_boundary(dict(status,**{key:value}),batch,receipt))


if __name__=='__main__':unittest.main()

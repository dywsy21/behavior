"""Exercise orchestration without importing a VLM, CUDA, or a simulator."""
import importlib.util
import io
from pathlib import Path
import sys
import time
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/rl')]
adapter=ModuleType('g05.rl.g05_adapter')
adapter.G05FlowAdapter=object; adapter.move=lambda x,device:x
old_adapter=sys.modules.get('g05.rl.g05_adapter'); old_learner=sys.modules.get('learner')
sys.modules['g05.rl.g05_adapter']=adapter
try:
    spec=importlib.util.spec_from_file_location('method_lifecycle_under_test',ROOT/'scripts/rl/method.py')
    method=importlib.util.module_from_spec(spec); spec.loader.exec_module(method)
finally:
    if old_adapter is None: sys.modules.pop('g05.rl.g05_adapter',None)
    else: sys.modules['g05.rl.g05_adapter']=old_adapter
    if old_learner is None: sys.modules.pop('learner',None)
    else: sys.modules['learner']=old_learner
base=method.Experiment
from g05.rl.flow_ppo import ControlBudget
from test_rl_automatic_curriculum import fixture


class LifecycleTests(unittest.TestCase):
    def test_automatic_course_does_not_wait_for_release_or_sleep(self):
        e=self.bare();spec,row=fixture()
        other=dict(spec,worker=1,gpu=3,instance=138)
        e.manifest.update(curriculum_admission='automatic',workers=[spec,other])
        e.prefix={0:1076,1:1076};e.connections={0:object(),1:object()};e.status=lambda **kw:None
        row['held']={'left':None,'right':'device'}
        with patch.object(method,'send'),patch.object(method,'recv',return_value=row),\
                patch.object(method,'save') as save,patch.object(method,'sha',return_value='image-hash'),\
                patch.object(method.time,'sleep',side_effect=AssertionError('must not wait')):
            e.review_curriculum(0)
        self.assertEqual(e.phase,'automatic_curriculum_passed')
        self.assertEqual(len(save.call_args_list),2)
        self.assertFalse(any('human_release' in str(call.args[0]) for call in save.call_args_list))

    def bare(self):
        e=method.MethodExperiment.__new__(method.MethodExperiment)
        e.manifest={'max_active_wall_seconds':500}; e.started=time.monotonic()
        e.pool='training'; e.training_deadline=time.monotonic()+100
        e.budget=ControlBudget(100); e.outstanding={}; e.connections={0:object()}
        e.latest={}; e.contexts={}; e.prepared={}; e.processes=[]; e.log=io.StringIO()
        return e

    def test_training_reserve_cutoff_does_not_stop_final_evaluation(self):
        e=self.bare(); e.training_deadline=time.monotonic()-1
        with self.assertRaises(method.TrainingCutoff): e.timecheck()
        e.pool='final'; e.timecheck()
        e.started=time.monotonic()-501
        with self.assertRaises(TimeoutError): e.timecheck()

    def test_no_wall_training_but_final_timeout_and_control_budget_remain(self):
        from g05.rl.time_limits import NO_TRAINING_WALL
        e=self.bare(); e.started=time.monotonic()-100000
        e.manifest.update(entry='method_dense',max_active_wall_seconds=None,
                          max_training_seconds=None,time_limit_override=NO_TRAINING_WALL)
        e.training_deadline=float('inf'); e.timecheck()
        e.pool='final'; e.final_deadline=time.monotonic()-1
        with self.assertRaisesRegex(TimeoutError,'final-evaluation'): e.timecheck()
        with self.assertRaises(RuntimeError): e.budget.reserve(101)

    def test_resumed_train_uses_cumulative_budget_batch_and_curriculum_history(self):
        from g05.rl.time_limits import NO_TRAINING_WALL
        e=self.bare(); e.manifest.update(entry='method_dense',max_active_wall_seconds=None,
            max_training_seconds=None,time_limit_override=NO_TRAINING_WALL,
            training_resume=dict(controls=8000,recent={'0':[True,True],'1':[]},zero_rewards=0,zero_updates=0),
            max_batches=3,max_new_actor_updates=2000,max_training_controls=8100,
            final_eval_reserved_controls=19344,episode_controls=16,curriculum_admission='automatic',
            learning_rates=dict(action_expert=1e-7,noise=1e-6),
            workers=[dict(instance=1,first_recorded_terminal=100),dict(instance=138,first_recorded_terminal=100)])
        e.budget=ControlBudget(100000); e.budget.used=8000
        e.batches=2; e.actor_updates=102; e.start_updates=94; e.prefix={0:10,1:10}
        e.stop_reason=None; e.last_checkpoint=None; e.status=lambda **kw:None
        e.actor_optimizer=type('Optimizer',(),{'param_groups':[{},{}]})()
        e.latest={0:dict(success=True,episode=0),1:dict(success=False,episode=0)}
        e.policy_controls={0:16,1:16}; checks=[]
        e.reset=lambda workers:self.fail('New sim already at reset; do not reset again')
        e.review_curriculum=lambda batch:checks.append(('review',batch))
        e.gates=lambda:checks.append(('gates',None))
        e.replay=lambda *a,**kw:setattr(e.budget,'used',e.budget.used+20)
        def rollout():
            e.budget.used+=32
            return [dict(split='train',rewards=[1.])],[],[],{0,1}
        e.rollout=rollout
        e.update=lambda *args:setattr(e,'actor_updates',e.actor_updates+1)
        e.checkpoint=lambda:None
        with patch.object(method,'save') as write,patch.object(method.torch,'save'):
            e.train()
        self.assertEqual(e.training_start_controls,0)
        self.assertEqual((e.batches,e.budget.used,e.actor_updates),(3,8052,103))
        self.assertEqual(checks,[('review',2),('gates',None)])
        self.assertEqual(e.prefix[0],0)  # prior two wins + this win progress the course
        self.assertEqual(write.call_args.args[1]['controls'],8052)
        self.assertEqual(write.call_args.args[1]['new_actor_updates'],9)

    def test_actual_not_reserved_controls_are_counted_and_no_double_dispatch(self):
        e=self.bare(); module=sys.modules.get(base.__module__)
        # The imported class retains the original globals after the temporary
        # dependency stub is removed; patch those globals directly.
        with patch.dict(base.issue.__globals__,{'send':lambda conn,row:None}):
            n=e.issue(0,[0]*16,'policy')
            self.assertEqual((n,e.budget.pending),(16,16))
            with self.assertRaises(ValueError): e.issue(0,[0],'policy')
        with patch.dict(base.collect_reply.__globals__,{'recv':lambda conn:dict(actual_controls=5)}):
            e.collect_reply(0,16)
        self.assertEqual((e.budget.used,e.budget.pending,e.outstanding),(5,0,{}))

    def test_failed_pool_with_unaccounted_controls_blocks_continuation(self):
        e=self.bare(); e.connections={}; e.budget.reserve(16)
        with patch.object(method,'save'):
            with self.assertRaisesRegex(RuntimeError,'close cleanly'): e.close_pool()
            e.close_pool(strict=False)


if __name__=='__main__': unittest.main()

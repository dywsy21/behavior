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


class LifecycleTests(unittest.TestCase):
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

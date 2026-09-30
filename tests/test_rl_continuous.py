import io
import importlib.util
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_rl_method_lifecycle as lifecycle
import test_rl_time_resume as time_resume
method=lifecycle.method
from g05.rl.continuous import AUTHORIZATION,CAPS,MIN_FREE_BYTES,continuous_mode,batch_indices,check_storage
from g05.rl.continuous_handoff import PRIOR_SOURCE,dependency_state,check_closed_pool,live_pids
from g05.rl.flow_ppo import ControlBudget
from g05.rl.dense_resume import resume_state
from g05.rl.dense_recipe import REWARD,validate_recipe
from g05.rl.time_limits import NO_TRAINING_WALL


def manifest():
    m=dict(entry='method_dense',training_limit_override=AUTHORIZATION,
        continuation_reason=AUTHORIZATION,time_limit_override=NO_TRAINING_WALL,
        max_training_seconds=None,max_active_wall_seconds=None,
        min_free_disk_bytes=MIN_FREE_BYTES,checkpoint_retention='retain_all_no_automatic_deletion',
        prior_evaluation_controls=19344,render_completion_retries=2,
        ae_precision='float32',curriculum_admission='automatic',bounded_backtracking=True,
        reset_candidate_lr_each_minibatch=True,ppo_epochs=4,critic_steps_per_batch=4,
        clip=.1,target_path_kl=.1,target_mean_path_kl=.02,bc_weight=.1,
        gamma_control=.9998,lambda_chunk=.95,episode_controls=1024,
        final_eval_reserved_controls=19344,final_eval_reserved_seconds=10800,
        training_resume=dict(batches=100,controls=400000,actor_updates=2200,critic_updates=416,
                             recent={'0':[],'1':[]},zero_rewards=4,zero_updates=4),
        reward=REWARD,learning_rates=dict(action_expert=1e-7,noise=1e-6,critic=1e-4),
        backtracking_scales=[1.,.5,.25,.125,.0625,.03125])
    m.update({k:None for k in CAPS})
    return m


class ContinuousTests(unittest.TestCase):
    def test_all_caps_removed_and_no_large_integer_substitute(self):
        m=manifest(); self.assertTrue(continuous_mode(m)); validate_recipe(m)
        self.assertEqual(list(islice(batch_indices(m,10**15),3)),[10**15,10**15+1,10**15+2])
        for key in CAPS:
            with self.assertRaises(ValueError): continuous_mode(dict(m,**{key:999999999}))
            missing=dict(m);del missing[key]
            with self.assertRaises(ValueError): continuous_mode(missing)
        with self.assertRaises(ValueError): validate_recipe(dict(m,bc_weight=0))
        with self.assertRaises(ValueError): continuous_mode(dict(m,training_limit_override='unknown'))

    def test_unlimited_control_ledger_still_reserves_exact_actual_and_pending(self):
        with self.assertRaises(ValueError): ControlBudget(None)
        with self.assertRaises(ValueError): ControlBudget(100,unlimited=True)
        b=ControlBudget(None,unlimited=True);b.used=10**15
        b.reserve(16);b.finish(16,5)
        self.assertEqual((b.used,b.pending),(10**15+5,0))
        with self.assertRaises(ValueError): b.finish(16,5)

    def test_storage_guard_does_not_delete_any_checkpoint(self):
        with patch('g05.rl.continuous.shutil.disk_usage',return_value=SimpleNamespace(free=107*(1<<30))):
            with self.assertRaisesRegex(RuntimeError,'no checkpoint deleted'): check_storage(manifest(),'.')
        with patch('g05.rl.continuous.shutil.disk_usage',return_value=SimpleNamespace(free=109*(1<<30))):
            check_storage(manifest(),'.')

    def test_resume_reconstructs_even_when_former_budget_or_stall_gate_is_exhausted(self):
        m,rows,receipt=time_resume.TimeResumeTests().fixture()
        m['max_batches']=2;m['max_training_controls']=8000
        with self.assertRaises(ValueError): resume_state(m,rows,receipt,8000)
        s=resume_state(m,rows,receipt,8000,continuation_authorization=AUTHORIZATION)
        self.assertEqual((s['batches'],s['actor_updates'],s['controls']),(2,102,8000))
        with self.assertRaises(ValueError):
            resume_state(m,rows,dict(receipt,actor_updates=101),8000,continuation_authorization=AUTHORIZATION)

    def test_actual_training_loop_crosses_old_limits_and_stall_gate(self):
        e=lifecycle.LifecycleTests().bare();e.manifest=manifest()
        e.manifest.update(episode_controls=16,workers=[dict(instance=i,first_recorded_terminal=100)
                                                     for i in (1,138)])
        e.budget=ControlBudget(None,unlimited=True);e.budget.used=400000
        e.batches=100;e.actor_updates=2200;e.start_updates=94;e.prefix={0:10,1:10}
        e.stop_reason=None;e.last_checkpoint=None;e.status=lambda **kw:None
        e.actor_optimizer=SimpleNamespace(param_groups=[{},{}]);e.latest={w:dict(success=False,episode=0) for w in (0,1)}
        e.policy_controls={0:16,1:16};e.reset=lambda workers:None
        e.review_curriculum=lambda batch:None;e.gates=lambda:None
        e.replay=lambda *a,**kw:setattr(e.budget,'used',e.budget.used+20)
        def rollout():
            if e.batches==105: raise RuntimeError('test-only external stop')
            e.budget.used+=32
            return [dict(split='train',rewards=[0.],official_rewards=[0.],shaping_rewards=[0.])],[],[],{0,1}
        e.rollout=rollout;e.update=lambda *a:None;e.checkpoint=lambda:None
        with patch.object(method,'save'),patch.object(method.torch,'save'),patch.object(method,'check_storage'):
            with self.assertRaisesRegex(RuntimeError,'test-only external stop'): e.train()
        self.assertEqual(e.batches,105)
        self.assertEqual(e.actor_updates,2200)
        self.assertIn('learning_stalled_warning',e.log.getvalue())
        self.assertIsNone(e.stop_reason)

    def test_running_dependency_does_not_read_final_or_launch_anything(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'supervisor.json').write_text(json.dumps(dict(source_commit=PRIOR_SOURCE,status='running')))
            self.assertFalse(dependency_state(p)['ready'])
            (p/'supervisor.json').write_text(json.dumps(dict(source_commit=PRIOR_SOURCE,status='failed')))
            with self.assertRaisesRegex(ValueError,'E3 failed'): dependency_state(p)

    def test_closed_pool_requires_exact_physical_count_and_all_exits(self):
        closed=dict(clean=True,pending_controls=0,exits=[0,0],reported_controls=[100,200],ledger_controls=300)
        self.assertEqual(check_closed_pool(closed,[100,200]),300)
        for field,value in [('clean',False),('pending_controls',16),('exits',[0,-15]),('ledger_controls',299)]:
            with self.assertRaises(ValueError):check_closed_pool(dict(closed,**{field:value}),[100,200])

    def test_completed_dependency_requires_all_six_fixed_checkpoint_episodes(self):
        from g05.rl.protocol import paired_summary
        with TemporaryDirectory() as d:
            p=Path(d);rows=[]
            for variant,actor in [('parent_fp32',0),('rl_fp32',211)]:
                for instance in (301,302):
                    for seed in (17,23,41):
                        rows.append(dict(variant=variant,actor_updates=actor,ae_precision='float32',instance=instance,
                            policy_seed=seed,split='public_test',environment_seed=0,expert_prefix_controls=0,
                            control_limit=3224,controls=3224,success=False,terminated=False,truncated=False))
            result=dict(paired_summary(rows),checkpoint='/saved.pt',checkpoint_sha256='abc',actor_updates=211,controls=99344)
            records={'supervisor.json':dict(source_commit=PRIOR_SOURCE,status='completed',exit_code=0),
                'status.json':dict(phase='completed',pending_controls=0,actor_updates=211,controls=99344),
                'training_result.json':dict(checkpoint='/saved.pt',actor_updates=211,controls=80000),
                'frozen_final_selection.json':dict(checkpoint='/saved.pt',sha256='abc',eval_results_used=False),
                'result.json':result}
            for name,value in records.items():(p/name).write_text(json.dumps(value))
            def eval_rows(values):(p/'evaluations.jsonl').write_text('\n'.join(json.dumps(r) for r in values))
            eval_rows(rows);ready=dependency_state(p)
            self.assertTrue(ready['ready']);self.assertEqual(ready['evaluation_controls'],19344)
            eval_rows(rows[:-1])
            with self.assertRaises(ValueError):dependency_state(p)
            eval_rows([dict(r,actor_updates=210) if r['variant']=='rl_fp32' else r for r in rows])
            with self.assertRaises(ValueError):dependency_state(p)

    def handoff_module(self):
        path=Path(__file__).resolve().parents[1]/'scripts/rl/continuous_handoff.py'
        spec=importlib.util.spec_from_file_location('handoff_under_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_actual_watch_does_not_touch_gpu_or_prepare_while_old_run_is_live(self):
        h=self.handoff_module()
        with TemporaryDirectory() as d:
            p=Path(d);(p/'claim.json').write_text(json.dumps(dict(source_commit='test')))
            with patch.object(h,'HANDOFF',p),patch.object(h,'validate_paths'),patch.object(h,'commit',return_value='test'),\
                    patch.object(h,'dependency_state',return_value=dict(ready=False,reason='old running')),\
                    patch.object(h.time,'sleep',side_effect=RuntimeError('test interruption')),\
                    patch.object(h.subprocess,'check_output') as launch,patch.object(h.subprocess,'run') as prepare:
                with self.assertRaisesRegex(RuntimeError,'test interruption'):h.watch()
            launch.assert_not_called();prepare.assert_not_called()

    def test_arm_is_exclusive_and_refuses_duplicate_watchers(self):
        h=self.handoff_module()
        with TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(h,'HANDOFF',root/'handoff'),patch.object(h,'OUT',root/'out'),\
                    patch.object(h,'RUNTIME',root/'runtime'),patch.object(h,'validate_paths'),\
                    patch.object(h,'commit',return_value='test'),\
                    patch.object(h.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as spawn:
                h.arm()
                with self.assertRaises(FileExistsError):h.arm()
            self.assertEqual(spawn.call_count,1)


if __name__=='__main__':unittest.main()

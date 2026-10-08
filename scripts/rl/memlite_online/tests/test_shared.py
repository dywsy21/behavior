"""Real multi-process/Gloo gradients, Adam, rollback and empty-rank contracts."""
# ruff: noqa: E402 -- standalone tests import the repository's adapter directory.
from contextlib import nullcontext
import ast
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'code'))
from direct_a4_flow import ValueHead
from shared_ppo import SharedFlowPPO
from synchronous import Collective, global_accept, state_digest, task_weights


class TinyPPO(SharedFlowPPO):
    def evaluate_prepared(self, batch, chains, old_log_probs, advantages, returns):
        from direct_ppo_core import ppo_clipped_policy_loss
        features = torch.tensor([[1., batch['x']]])
        new = self.policy.model.action_expert(features).reshape(-1)
        loss, metrics = ppo_clipped_policy_loss(new, old_log_probs.reshape(-1), advantages,
                                                clip_ratio=self.clip_ratio)
        value = self.critic(features)
        value_loss = .5 * (value-returns).square().mean()
        kl = float(metrics['approx_kl'])
        delta = float(self.actor_parameters[0].detach().abs().max())
        if self.test_mode == 'backtrack' and self.collective.rank == 1:
            kl = (delta * 1e7) ** 2 * .1
        if self.test_mode == 'reject' and delta:
            kl = 1.
        return loss + .5 * value_loss, dict(policy_loss=float(loss.detach()),
            value_loss=float(value_loss.detach()), ratio_mean=float(metrics['ratio_mean']),
            approx_kl=kl, clip_fraction=float(metrics['clip_fraction']))


def worker(rank, root, mode):
    torch.set_num_threads(1)
    c = Collective.initialize(rank, 2, 'file://' + str(Path(root)/'rendezvous'), backend='gloo', timeout=45)
    try:
        if mode == 'gradient':
            parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
            rows = [(torch.tensor([2., -1.]), 1.), (torch.tensor([1., 3.]), 2.)] if rank == 0 else [
                (torch.tensor([-2., 4.]), 4.)]
            for x, weight in rows:
                ((parameter * x).sum().square() * weight).backward()
            c.gradients([parameter], denominator=7.)
            reference = torch.tensor([1., 2.], requires_grad=True)
            total = sum(((reference*x).sum().square()*w) for x, w in [
                (torch.tensor([2., -1.]), 1.), (torch.tensor([1., 3.]), 2.), (torch.tensor([-2., 4.]), 4.)]) / 7.
            total.backward()
            torch.testing.assert_close(parameter.grad, reference.grad)
            c.same(state_digest(parameter.grad), 'gradient')
            result = dict(ok=True)
        elif mode == 'error':
            try:
                c.check(ValueError('local fault') if rank else None, 'test')
                raise AssertionError('peer error did not propagate')
            except RuntimeError as error:
                assert 'local fault' in str(error)
                result = dict(ok=True)
        else:
            torch.manual_seed(12)
            trainer = TinyPPO.__new__(TinyPPO)
            policy = torch.nn.Module()
            policy.model = torch.nn.Module()
            policy.model.action_expert = torch.nn.Linear(2, 1, bias=False)
            policy.model.action_expert.weight.data.zero_()
            trainer.policy = policy
            trainer.actor_parameters = list(policy.parameters())
            trainer.actor_names = ['model.action_expert.weight']
            trainer.actor_optimizer = torch.optim.AdamW(trainer.actor_parameters, lr=1e-7, weight_decay=0.)
            trainer.critic = ValueHead(2)
            trainer.critic_optimizer = torch.optim.AdamW(trainer.critic.parameters(), lr=1e-4, weight_decay=0.)
            trainer.output_dir = Path(root)/f'rank{rank}'/'checkpoints'
            trainer.output_dir.mkdir(parents=True)
            trainer.collective = c
            trainer.task_weight_map = {'a': 1., 'b': 2.}
            trainer.nominal_actor_lr = 1e-7
            trainer.critic_lr = 1e-4
            trainer.target_kl = .02
            trainer.max_clip_fraction = .25
            trainer.clip_ratio = .2
            trainer.transition_std = .02
            trainer.reward_protocol = 'shared_terminal_q_v1'
            trainer.update_count = trainer.actor_update_count = 0
            trainer.checkpoint_interval = 10
            trainer.shared_checkpoint_state = trainer.last_shared_receipt = None
            trainer.initial_update = 0
            trainer.test_mode = mode
            count = (0 if mode == 'empty' else 3) if rank == 0 else 1
            trainer.experiences = {i: dict(observation={'x': float(i + rank)}, local_subgoal={},
                chain=torch.zeros(2, 1, 1), old_step_log_prob=torch.zeros(1), old_value=0.,
                task_identity='a' if rank == 0 else 'b', policy_version=0) for i in range(count)}
            trainer.next_experience_id = count
            inferencer = types.SimpleNamespace(prepare_training_batch=lambda obs, goals: obs[0],
                _branch_context=nullcontext, episode_metadata=[])
            initial = state_digest(dict(actor=policy.state_dict(), adam=trainer.actor_optimizer.state_dict(),
                                        critic=trainer.critic.state_dict()))
            advantages = [0. if mode == 'zero_global' or rank == 0 and mode == 'zero_local' else 1.] * count
            with patch('torch.cuda.get_rng_state', return_value=torch.get_rng_state()):
                try:
                    result = trainer.update(inferencer, list(range(count)), advantages, [2.] * count)
                    if mode == 'reject':
                        raise AssertionError('Rejection expected')
                    assert result['update'] == 1 and result['actor_updated'] == (mode != 'zero_global')
                    assert result['global_experiences'] == count + (1 if rank == 0 else (0 if mode == 'empty' else 3))
                    assert not trainer.experiences
                    if mode == 'backtrack':
                        assert len(result['backtracking']) > 1
                        assert result['accepted_actor_lr'] < 1e-7
                    if mode == 'resume':
                        expected = state_digest(dict(actor=policy.state_dict(), adam=trainer.actor_optimizer.state_dict()))
                        with torch.no_grad():
                            trainer.actor_parameters[0].add_(1.)
                        original_load = torch.load
                        def cpu_load(*args, **kwargs):
                            kwargs['map_location'] = 'cpu'
                            return original_load(*args, **kwargs)
                        with patch('torch.load', side_effect=cpu_load), patch.object(torch.nn.Module, 'cuda', lambda s: s), \
                                patch('torch.cuda.set_rng_state'):
                            receipt = trainer.load_checkpoint(Path(result['checkpoint']['latest']))
                        assert receipt['environment_resume'] == 'reset_all_envs_discard_uncommitted_rollouts'
                        assert expected == state_digest(dict(actor=policy.state_dict(), adam=trainer.actor_optimizer.state_dict()))
                except RuntimeError:
                    if mode != 'reject':
                        raise
                    assert initial == state_digest(dict(actor=policy.state_dict(),
                        adam=trainer.actor_optimizer.state_dict(), critic=trainer.critic.state_dict()))
                    assert trainer.update_count == 0
                    result = dict(rejected_restored=True)
            c.same(state_digest(dict(actor=policy.state_dict(), optimizer=trainer.actor_optimizer.state_dict(),
                critic=trainer.critic.state_dict(), critic_optimizer=trainer.critic_optimizer.state_dict())), 'final')
        (Path(root)/f'result{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class SharedContracts(unittest.TestCase):
    def run_two(self, mode):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(worker, args=(root, mode), nprocs=2, join=True)
            rows = [json.loads((Path(root)/f'result{rank}.json').read_text()) for rank in range(2)]
            return rows

    def test_weighted_gradients_match_one_global_batch(self):
        self.run_two('gradient')

    def test_peer_failure_propagates_without_deadlock(self):
        self.run_two('error')

    def test_shared_adam_and_checkpoint_identity(self):
        rows = self.run_two('normal')
        self.assertEqual(rows[0]['replica_identity'], rows[1]['replica_identity'])
        self.assertEqual(rows[0]['checkpoint']['sha256'], rows[1]['checkpoint']['sha256'])

    def test_empty_rank_still_participates(self):
        self.run_two('empty')

    def test_zero_local_advantage_receives_global_update(self):
        self.run_two('zero_local')

    def test_zero_global_advantage_only_updates_critic(self):
        rows = self.run_two('zero_global')
        self.assertFalse(rows[0]['actor_updated'])
        self.assertEqual(rows[0]['accepted_actor_lr'], 0.)

    def test_exact_shared_checkpoint_resume(self):
        self.run_two('resume')

    def test_one_bad_rank_forces_all_rank_backtracking(self):
        rows = self.run_two('backtrack')
        self.assertEqual(rows[0]['accepted_actor_lr'], rows[1]['accepted_actor_lr'])

    def test_rejection_restores_all_adam_and_parameters(self):
        self.run_two('reject')

    def test_unequal_group_and_horizon_weights(self):
        groups = [{'tasks': [{'task': 'a', 'official_horizon_steps': 9}]},
                  {'tasks': [{'task': 'b', 'official_horizon_steps': 19},
                             {'task': 'c', 'official_horizon_steps': 39}]}]
        weights = task_weights(groups)
        for name, nominal_probability in [('a', .5), ('b', 1/6), ('c', 1/3)]:
            self.assertAlmostEqual(weights[name] * nominal_probability, 1/3)
        with self.assertRaises(ValueError):
            task_weights([groups[0], groups[0]])

    def test_global_average_cannot_hide_bad_group(self):
        self.assertFalse(global_accept(dict(mean_approx_kl=.01, clip_fraction=.01, ratio_mean=1.),
            [dict(experiences=1, mean_approx_kl=.1, clip_fraction=.01, ratio_mean=1.)],
            target_kl=.02, max_clip_fraction=.25))

    def test_actual_shared_task_reset_and_observation_binding(self):
        class Ledger:
            def __init__(self, task):
                self.memory = json.dumps(dict(task_name=task, issued_command_history=[], verified_world_facts=[]))
                self.revision = 0
        begin = next(n for n in ast.walk(ast.parse((SOURCE/'code/stage1_engine.py').read_text()))
                     if isinstance(n, ast.FunctionDef) and n.name == 'begin')
        namespace = dict(torch=torch, json=json, PlannerLedger=Ledger,
            Path=lambda path: types.SimpleNamespace(read_text=lambda: '\n'.join(
                json.dumps(dict(task_name=t)) for t in ['task_a', 'task_b'])))
        exec(compile(ast.Module(body=[begin], type_ignores=[]), 'real-base-begin', 'exec'), namespace)
        base = type('Stage1Engine', (), dict(begin=namespace['begin'], prepare=lambda *a: 'prepared'))
        cls_node = next(n for n in ast.parse((SOURCE/'code/shared_engine.py').read_text()).body if isinstance(n, ast.ClassDef))
        batch_module = SOURCE.parents[1]/'eval/memlite_sft100/batch_core.py'
        batch_namespace = {}
        exec(compile(batch_module.read_text(), str(batch_module), 'exec'), batch_namespace)
        namespace = dict(torch=torch, json=json, Stage1Engine=base, validate_indices=batch_namespace['validate_indices'])
        exec(compile(ast.Module(body=[cls_node], type_ignores=[]), 'real-shared-engine', 'exec'), namespace)
        engine = namespace['SharedStage1Engine'].__new__(namespace['SharedStage1Engine'])
        engine.trainer = types.SimpleNamespace(experiences={}, update_count=7)
        engine.collective = types.SimpleNamespace(rank=0)
        engine.shared_manifest = dict(groups=[dict(tasks=[dict(task=t, train_instances=[1, 2]) for t in ['task_a', 'task_b']])])
        engine.session_generation = 0
        with tempfile.TemporaryDirectory() as root:
            engine.output = Path(root)/'checkpoints'
            for task in ['task_a', 'task_b', 'task_a']:
                metadata = [dict(task=task, split='train', gpu=0, instance_id=i, episode_id=f'{task}:{i}') for i in [1, 2]]
                engine.begin(task, 2, 17, metadata)
                obs = [dict(task_identity=task, episode_id=metadata[1]['episode_id'])]
                engine.validate_observations(obs, [1])
                with self.assertRaises(ValueError):
                    engine.validate_observations(obs, [0])
                with self.assertRaises(ValueError):
                    engine.validate_observations([dict(task_identity='other', episode_id=metadata[1]['episode_id'])], [1])
                valid = dict(task_name=engine.task, memlite_branch='low')
                self.assertEqual(engine.prepare('low', obs, [valid]), 'prepared')
                with self.assertRaises(ValueError):
                    engine.prepare('low', obs, [dict(valid, memlite_branch='high')])
                engine.slots[1]['ledger'].memory = 'stale task memory'
            with self.assertRaises(ValueError):
                engine.begin('task_a', 2, 17, [dict(metadata[0], instance_id=301), metadata[1]])

    def test_training_monitor_never_counts_partial_or_repeats_rows(self):
        from shared_monitor import TrainingMonitor
        with tempfile.TemporaryDirectory() as root:
            job = Path(root)
            cycle = job/'collectors/gpu_0/cycle_00000_task_a'
            result_dir = cycle/'eval_task_a/json'
            result_dir.mkdir(parents=True)
            record = dict(task='task_a', episode={'episode_id': 'a'}, official_q=.2,
                          episode_control_step=1, skill_potential=.5, official_success=False, task_terminal=False)
            (cycle/'reward_steps.jsonl').write_text(json.dumps(record)+'\n'+json.dumps(record))
            (result_dir/'result.json').write_text(json.dumps(dict(success=False, q_score={'final': .2})))
            monitor = TrainingMonitor(job)
            first = monitor.refresh()
            self.assertEqual(first['controls'], 1)
            self.assertIsNone(first['completed_only_sr'])
            self.assertEqual(monitor.refresh()['controls'], 1)
            with (cycle/'reward_steps.jsonl').open('a') as stream:
                stream.write('\n')
            (cycle/'task_a.completed').touch()
            final = monitor.refresh()
            self.assertEqual(final['controls'], 2)
            self.assertEqual(final['completed_episodes'], 1)
            self.assertEqual(final['macro_q_covered'], .2)


if __name__ == '__main__':
    unittest.main()

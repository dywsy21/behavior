import copy
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/rl')]
from g05.rl.recovery import remaining_evaluation_budget, validate_baseline, count_physical_steps

adapter = ModuleType('g05.rl.g05_adapter')
adapter.G05FlowAdapter = object
adapter.move = lambda x, device: x
names = ('g05.rl.g05_adapter', 'learner', 'method')
old = {name: sys.modules.get(name) for name in names}
sys.modules['g05.rl.g05_adapter'] = adapter
try:
    spec = importlib.util.spec_from_file_location('final_recovery_under_test', ROOT/'scripts/rl/final_eval.py')
    final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final)
finally:
    for name, value in old.items():
        if value is None: sys.modules.pop(name, None)
        else: sys.modules[name] = value


def baseline():
    return [dict(variant=name, instance=i, policy_seed=s, split='public_test',
                 environment_seed=0, expert_prefix_controls=0, control_limit=3224,
                 controls=3224, success=False, terminated=False, truncated=False,
                 actor_updates=0, ae_precision=precision)
            for name, seeds, precision in [('parent_fp32', (17, 23, 41), 'float32'),
                                            ('parent_bf16', (17,), 'bfloat16')]
            for i in (301, 302) for s in seeds]


class RecoveryTests(unittest.TestCase):
    def budget(self, **status_changes):
        status = dict(phase='failed', controls=48958, pending_controls=0, seconds=29696.8353)
        status.update(status_changes)
        return remaining_evaluation_budget(dict(max_controls=100000, max_active_wall_seconds=43200),
            status, dict(status='failed', seconds=29702.7443), 23166, 25792)

    def test_original_budget_is_debited_and_final_matrix_only(self):
        b = self.budget()
        self.assertEqual(b['prior_controls'], 48958)
        self.assertEqual(b['prior_active_seconds'], 29703)
        self.assertEqual(b['max_controls'], 19344)
        self.assertEqual(b['max_active_wall_seconds'], 10800)
        self.assertLess(b['prior_active_seconds']+b['max_active_wall_seconds']+40, 43200)

    def test_pending_or_wrong_physical_ledger_rejected(self):
        for bad in ({'pending_controls': 16}, {'controls': 48959}, {'phase': 'collecting'}):
            with self.assertRaises(ValueError): self.budget(**bad)

    def test_exhausted_time_or_nonfinite_time_rejected(self):
        for t in (41000, float('inf'), float('nan')):
            with self.assertRaises(ValueError): self.budget(seconds=t)

    def test_complete_baseline_retains_precision_anchor_separately(self):
        self.assertEqual(validate_baseline(baseline()), 25792)

    def test_missing_duplicate_final_or_invalid_baseline_rejected(self):
        good = baseline()
        cases = [good[:-1], good+[good[0]], [dict(good[0], variant='rl_fp32')]+good[1:]]
        for change in ({'controls': 50}, {'expert_prefix_controls': 1}, {'actor_updates': 94},
                       {'success': True}, {'ae_precision': 'bfloat16'}, {'invalid': True}):
            cases.append([dict(good[0], **change)]+good[1:])
        for rows in cases:
            with self.assertRaises(ValueError): validate_baseline(rows)

    def test_physical_steps_require_contiguity_and_scope(self):
        rows = [dict(control=1, episode=0, episode_control=1, phase='policy'),
                dict(control=2, episode=1, episode_control=1, phase='expert_prefix')]
        self.assertEqual(count_physical_steps(rows, allowed_phases={'policy', 'expert_prefix'}), 2)
        for change in ({'control': 3}, {'episode_control': 2}, {'episode': 2}, {'phase': 'eval_rl_fp32'}):
            with self.assertRaises(ValueError):
                count_physical_steps([rows[0], dict(rows[1], **change)], allowed_phases={'policy', 'expert_prefix'})

    def test_recovery_cannot_train_replay_or_write_weights(self):
        e = final.FinalEvaluation.__new__(final.FinalEvaluation)
        for name in ('train', 'update', 'rollout', 'replay', 'prepare_bc', 'checkpoint', 'gates'):
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                getattr(e, name)()
        for pool in ('baseline', 'training', None):
            with self.assertRaises(ValueError): e.spawn_pool(pool)
        with self.assertRaises(ValueError): e.restore_delta('x', 'y', optimizer=True)

    def test_real_restore_precedes_spawn_and_same_fixed_matrix(self):
        e = final.FinalEvaluation.__new__(final.FinalEvaluation)
        e.manifest = dict(audit_input_sha256={}, final_checkpoint='/tmp/test-delta.pt',
            final_checkpoint_sha256='saved-hash', prior_controls=48958, prior_active_seconds=29703,
            continued_from='old')
        e.log = io.StringIO(); e.eval_log = io.StringIO()
        e.evaluations = baseline()
        e.started = final.time.monotonic()
        e.budget = type('Budget', (), {'used': 19344})()
        events = []
        e.load = lambda: events.append('parent')
        def restore(path, expected, *, optimizer):
            self.assertEqual((str(path), expected, optimizer), ('/tmp/test-delta.pt', 'saved-hash', False))
            events.append('saved_delta'); e.actor_updates = 94; e.critic_updates = 16
        e.restore_delta = restore
        e.spawn_pool = lambda pool: events.append(pool)
        def evaluate(variants):
            self.assertEqual(variants, [('rl_fp32', [17, 23, 41], 'float32')])
            events.append('matrix')
            e.evaluations += [dict(r, variant='rl_fp32', actor_updates=94) for r in baseline() if r['variant']=='parent_fp32']
        e.evaluate = evaluate
        e.close_pool = lambda **kw: events.append('close')
        e.status = lambda **kw: None
        with patch.object(final, 'save') as save:
            e.run()
            result = save.call_args_list[0].args[1]
            self.assertEqual((result['before_successes'], result['after_successes']), (0, 0))
            self.assertEqual(result['cumulative_controls'], 68302)
            self.assertEqual(result['new_training_updates'], 0)
        self.assertEqual(events, ['parent', 'saved_delta', 'final', 'matrix', 'close', 'close'])


if __name__ == '__main__': unittest.main()

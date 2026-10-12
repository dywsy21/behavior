import json
from pathlib import Path
import tempfile
import types
import unittest

from behavior_champion.runner import run_updates, validate_launch


class FakeEngine:
    def __init__(self, clock, *, completed=0, bad=False):
        self.completed, self.clock, self.bad = completed, clock, bad
        self.saves, self.evals, self.exports = [], [], []
    def step(self, step):
        assert step == self.completed
        self.completed += 1; self.clock[0] += 10
        return {'loss': float('nan') if self.bad else 1 / self.completed}
    def save(self):
        self.saves.append(self.completed)
        return dict(completed_updates=self.completed, path='fake')
    def evaluate(self):
        self.evals.append(self.completed); return {'eval/fm': .5}
    def export(self, receipt):
        self.exports.append(self.completed); return {'step': self.completed}


class Tracker:
    def __init__(self): self.logs, self.finishes = [], []
    def log(self, metrics, *, completed): self.logs.append((completed, metrics))
    def finish(self, status, **kwargs): self.finishes.append((status, kwargs))


class RunnerTests(unittest.TestCase):
    def run_loop(self, directory, engine, tracker, stop, *, prior=0, limit=100):
        return run_updates(engine, tracker, directory, updates=5, max_seconds=limit,
                           checkpoint_every=2, eval_every=2, stop=stop, prior_seconds=prior,
                           started=0, clock=lambda: engine.clock[0])

    def test_registered_pending_recipe_refuses_formal_before_execution(self):
        root = Path(__file__).resolve().parents[1]
        recipe = json.loads((root / 'configs/champion2026/shared100_v1.json').read_text())
        self.assertFalse(recipe['formal_launch_approved'])
        with self.assertRaises(PermissionError): validate_launch(recipe, engineering=False)
        validate_launch(recipe, engineering=True)
        with self.assertRaises(ValueError): validate_launch(dict(recipe, updates=8001), engineering=True)

    def test_complete_saves_evaluates_and_exports_exact_final_cursor(self):
        engine, tracker = FakeEngine([0]), Tracker()
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_loop(directory, engine, tracker, types.SimpleNamespace(requested=False))
            self.assertEqual(result['status'], 'completed_target_updates')
            self.assertEqual(engine.saves, [2, 4, 5])
            self.assertEqual(engine.evals, [0, 2, 4, 5])
            self.assertEqual(engine.exports, [5])
            events = [json.loads(line) for line in (Path(directory) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([r['completed_updates'] for r in events if r['event'] == 'optimizer_update'], [1,2,3,4,5])

    def test_wall_budget_is_cumulative_across_resume(self):
        engine, tracker = FakeEngine([0], completed=2), Tracker()
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_loop(directory, engine, tracker, types.SimpleNamespace(requested=False), prior=90)
            self.assertEqual(result['status'], 'saved_at_cumulative_time_limit')
            self.assertEqual(engine.completed, 3)
            self.assertEqual(engine.saves, [3])
            self.assertEqual(engine.evals, [])
            self.assertEqual(engine.exports, [])

    def test_signal_saves_without_another_update_or_eval(self):
        engine, tracker = FakeEngine([0], completed=2), Tracker()
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_loop(directory, engine, tracker, types.SimpleNamespace(requested=True))
            self.assertEqual(result['status'], 'saved_on_signal')
            self.assertEqual(engine.completed, 2)
            self.assertEqual(engine.saves, [2])
            self.assertEqual(engine.evals, [])

    def test_nonfinite_never_saved_as_a_resumable_state(self):
        engine, tracker = FakeEngine([0], bad=True), Tracker()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.run_loop(directory, engine, tracker, types.SimpleNamespace(requested=False))
            self.assertEqual(engine.saves, [])
            self.assertTrue((Path(directory) / 'failure.json').exists())
            self.assertFalse((Path(directory) / 'latest-checkpoint.json').exists())
            self.assertTrue(tracker.finishes[-1][1]['error'])


if __name__ == '__main__': unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / 'tools/audit_learning_signal.py'
spec = importlib.util.spec_from_file_location('audit_learning_signal', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class LearningSignalAuditTests(unittest.TestCase):
    def test_partial_jsonl_line_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            path.write_bytes(b'{"a":1}\n{"a":2}')
            self.assertEqual(list(audit.lines(path)), [{'a': 1}])

    def test_stats_and_empty_input(self):
        self.assertIsNone(audit.stats([]))
        self.assertEqual(audit.stats([4, 1, 3, 2])['median'], 2.5)

    def test_end_to_end_terminal_and_intent_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job/'manifest.json').write_text(json.dumps({'source_commit': 'test'}))
            cycle = job/'collectors/gpu_0/cycle_00000_test'
            policy = job/'policies/gpu_0'
            cycle.mkdir(parents=True)
            (policy/'checkpoints').mkdir(parents=True)
            episode = {'episode_id': 'test:env0', 'instance_id': 1}
            rows = [dict(episode=episode, task='test', episode_control_step=i+1,
                official_q=q, official_success=False, skill_potential=.2,
                task_terminal=i==2, shaping=0., terminal_q_reward=10*q if i==2 else 0.,
                reward=10*q if i==2 else 0., experience_id=i)
                for i, q in enumerate([0., .5, .5])]
            (cycle/'reward_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            event = dict(active_skills_semantic_json='[{"verb":"GRASP"}]',
                         decision='EXECUTE', previous_outcome='UNKNOWN')
            plans = [dict(episode=episode, control_step=i, event=event) for i in [0, 2]]
            (policy/'planner_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plans))
            result = dict(instance_id=1, steps=3, q_score={'final': .5}, success=False,
                          time={'normalized_time': 2/3})
            (cycle.parent/'status.json').write_text(json.dumps({'cycles': [
                {'task': 'test', 'official_episodes': [result]}]}))
            update = dict(accepted_actor_lr=1e-7, before={'mean_approx_kl': 0, 'value_loss': .1},
                          post_update={'mean_approx_kl': .001, 'clip_fraction': .01})
            (policy/'checkpoints/updates.jsonl').write_text(json.dumps(update)+'\n')
            result = audit.audit(job)
            self.assertEqual(result['totals']['controls'], 3)
            self.assertEqual(result['summary']['terminal_q_positive_chunks'], 1)
            self.assertEqual(result['summary']['completed_single_skill_episodes'], 1)
            self.assertEqual(result['summary']['completed_intent_static_fraction']['median'], 1.)
            self.assertEqual(result['summary']['episodes_q_increase'], 1)
            self.assertEqual(result['summary']['completed_episodes_lost_all_q'], 0)
            self.assertEqual(result['summary']['successes'], 0)


if __name__ == '__main__':
    unittest.main()

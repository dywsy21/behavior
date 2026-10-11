from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from behavior_champion.tracking import OnlineRun


class TrackingTests(unittest.TestCase):
    def test_online_exact_identity_and_optimizer_step_is_not_wandb_append_step(self):
        calls = []
        run = types.SimpleNamespace(entity='team', project='project', id='run-id', offline=False,
                                    url='https://example.invalid/run-id', summary={})
        run.define_metric = lambda *a, **kw: calls.append(('define', a, kw))
        run.log = lambda *a, **kw: calls.append(('log', a, kw))
        run.finish = lambda **kw: calls.append(('finish', kw))
        def init(**kwargs):
            self.assertEqual(kwargs['mode'], 'online')
            self.assertEqual(kwargs['resume'], 'must')
            self.assertEqual(kwargs['id'], 'run-id')
            return run
        fake = types.SimpleNamespace(init=init, Settings=lambda **kw: kw)
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {'wandb': fake}), patch.dict('os.environ'):
            credential = Path(root) / 'test-only.key'; credential.write_text('test-only-not-a-real-key')
            credential.chmod(0o600)
            tracker = OnlineRun(Path(root) / 'logs', credential_file=credential, entity='team', project='project',
                name='test', config={}, resume_id='run-id', resume_update=2000)
            tracker.log({'train/loss': .5}, completed=2001)
            receipt = tracker.finish('completed', completed=2001)
            self.assertEqual(receipt['run_id'], 'run-id')
            for kind, *rest in calls:
                if kind == 'log':
                    self.assertNotIn('step', rest[-1])
                    self.assertNotIn('test-only-not-a-real-key', str(rest))

    def test_insecure_credential_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as root:
            credential = Path(root) / 'test-only.key'; credential.write_text('dummy'); credential.chmod(0o644)
            with self.assertRaises(ValueError):
                OnlineRun(Path(root) / 'logs', credential_file=credential, entity='team', project='project',
                          name='test', config={})


if __name__ == '__main__': unittest.main()

"""CPU-only launch contract; no simulator imports or GPU processes."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/rl'), str(ROOT/'src')]
import rtx_speed as speed


class RTXLaunchTests(unittest.TestCase):
    def test_workers_inherit_broad_mask_before_controller_is_pinned(self):
        mask = set(range(24))
        events = []

        def popen(*args, **kwargs):
            events.append(('spawn', set(mask)))
            return SimpleNamespace(pid=100+len(events))

        def pin(pid, new_mask):
            self.assertEqual(pid, 0)
            mask.clear(); mask.update(new_mask)
            events.append(('pin', set(mask)))

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(speed, 'OUT', Path(folder)), \
                patch.object(speed.os, 'sched_getaffinity', side_effect=lambda _: set(mask)), \
                patch.object(speed.os, 'sched_setaffinity', side_effect=pin), \
                patch.object(speed, 'sim_env', side_effect=lambda _: {}), \
                patch.object(speed.subprocess, 'Popen', side_effect=popen):
            children = []
            speed.spawn_simulators(children, SimpleNamespace(address=('127.0.0.1',1234)), b'test')
            self.assertEqual(len(children), 2)
            self.assertEqual(events, [('spawn', set(range(24))),
                                      ('spawn', set(range(24))), ('pin', {16,17})])

    def test_missing_worker_cores_rejected_before_runtime_or_process(self):
        with patch.object(speed.os, 'sched_getaffinity', return_value={16,17}), \
                patch.object(speed, 'sim_env') as env, \
                patch.object(speed.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'CPU partition'):
                speed.spawn_simulators([], None, b'test')
            env.assert_not_called(); popen.assert_not_called()

    def test_partial_launch_remains_owned_for_controller_cleanup(self):
        first = SimpleNamespace(pid=101)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(speed, 'OUT', Path(folder)), \
                patch.object(speed.os, 'sched_getaffinity', return_value=set(range(24))), \
                patch.object(speed.os, 'sched_setaffinity') as pin, \
                patch.object(speed, 'sim_env', side_effect=lambda _: {}), \
                patch.object(speed.subprocess, 'Popen', side_effect=[first, OSError('spawn failed')]):
            children = []
            with self.assertRaises(OSError):
                speed.spawn_simulators(children, SimpleNamespace(address=('127.0.0.1',1234)), b'test')
            self.assertEqual(children, [first])
            pin.assert_not_called()


if __name__ == '__main__': unittest.main()

import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/semantic_robot'))
from shared_og_startup import private_og_startup
import shared_og_startup as bridge


class SharedOGStartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'simulator.py'; self.source.write_bytes(b'frozen source')
        self.pairs = {}
        for name in ('experience.kit', 'icon.png'):
            src, dst = self.root / ('source_' + name), self.root / name
            src.write_bytes(name.encode()); dst.write_bytes(name.encode())
            self.pairs[src, dst] = hashlib.sha256(name.encode()).hexdigest()
        self.experience = self.root / 'experience.kit'
        self.copy = Mock(side_effect=AssertionError('No real copy permitted'))
        self.original_factory = Mock(side_effect=AssertionError('Registered factory must be used'))
        self.module = SimpleNamespace(__file__=str(self.source), og=SimpleNamespace(app=None),
            shutil=SimpleNamespace(copyfile=self.copy, unrelated='unchanged'),
            lazy=SimpleNamespace(isaacsim=SimpleNamespace(SimulationApp=self.original_factory, other='kept'), other=42))
        self.config = {'headless': True, 'multi_gpu': False, 'active_gpu': 3, 'physics_gpu': 3}
        self.factory = Mock(return_value='bounded_app')
        self.module._launch_app = self.original_launch

    def original_launch(self):
        for src, dst in self.pairs: self.module.shutil.copyfile(src, dst)
        self.assertEqual(self.module.shutil.unrelated, 'unchanged')
        self.assertEqual(self.module.lazy.other, 42)
        self.assertEqual(self.module.lazy.isaacsim.other, 'kept')
        return self.module.lazy.isaacsim.SimulationApp(self.config, experience=str(self.experience))

    def context(self):
        return private_og_startup(self.module, source_sha256=hashlib.sha256(b'frozen source').hexdigest(),
            experience=self.experience, copy_bindings=self.pairs, construct=self.factory)

    def test_exact_original_route_and_restoration_without_install_writes(self):
        before = self.module.shutil, self.module.lazy, self.module._launch_app
        with self.context() as receipt:
            self.assertEqual(self.module._launch_app(), 'bounded_app')
            self.assertEqual((self.module.shutil, self.module.lazy), before[:2])
            with self.assertRaises(RuntimeError): self.module._launch_app()
        self.assertEqual((self.module.shutil, self.module.lazy, self.module._launch_app), before)
        self.assertEqual(receipt['app_constructions'], 1)
        self.assertEqual(len(receipt['verified_copy_noops']), 2)
        self.copy.assert_not_called(); self.original_factory.assert_not_called()
        self.factory.assert_called_once_with()

    def test_changed_source_resource_or_live_app_rejected_before_constructor(self):
        mutations = ('source', 'destination', 'live')
        for what in mutations:
            with self.subTest(what=what):
                old = self.source.read_bytes(), self.experience.read_bytes()
                if what == 'source': self.source.write_bytes(b'changed')
                if what == 'destination': self.experience.write_bytes(b'changed')
                if what == 'live': self.module.og.app = object()
                try:
                    with self.assertRaises(ValueError):
                        with self.context(): pass
                finally:
                    self.source.write_bytes(old[0]); self.experience.write_bytes(old[1]); self.module.og.app = None
        self.factory.assert_not_called(); self.copy.assert_not_called()

    def test_late_resource_change_unknown_copy_and_repeated_copy_fail_closed(self):
        for mode in ('late_change', 'unknown', 'repeat'):
            with self.subTest(mode=mode):
                def bad_launch():
                    src, dst = next(iter(self.pairs))
                    if mode == 'late_change': dst.write_bytes(b'changed')
                    if mode == 'unknown': return self.module.shutil.copyfile(src, self.root/'unknown')
                    self.module.shutil.copyfile(src, dst)
                    if mode == 'repeat': self.module.shutil.copyfile(src, dst)
                original_shutil, original_lazy = self.module.shutil, self.module.lazy
                self.module._launch_app = bad_launch
                try:
                    with patch.object(bridge, '_CONSUMED_PATHS', set()), self.context():
                        with self.assertRaises(ValueError): self.module._launch_app()
                finally:
                    src, dst = next(iter(self.pairs)); dst.write_bytes(src.read_bytes())
                self.assertIs(self.module.shutil, original_shutil); self.assertIs(self.module.lazy, original_lazy)
        self.factory.assert_not_called(); self.copy.assert_not_called()

    def test_wrong_gpu_config_experience_and_second_factory_rejected(self):
        for key, value in (('active_gpu', 0), ('physics_gpu', True), ('multi_gpu', True), ('headless', 1)):
            with self.subTest(key=key):
                old = self.config.copy(); self.config[key] = value
                try:
                    with patch.object(bridge, '_CONSUMED_PATHS', set()), self.context():
                        with self.assertRaises(ValueError): self.module._launch_app()
                finally: self.config = old
        self.factory.assert_not_called()
        def bad_experience():
            for src, dst in self.pairs: self.module.shutil.copyfile(src, dst)
            return self.module.lazy.isaacsim.SimulationApp(self.config, experience=str(self.root/'wrong.kit'))
        self.module._launch_app = bad_experience
        with patch.object(bridge, '_CONSUMED_PATHS', set()), self.context():
            with self.assertRaises(ValueError): self.module._launch_app()
        self.factory.assert_not_called()
        def repeated_factory():
            self.original_launch()
            return self.module.lazy.isaacsim.SimulationApp(self.config, experience=str(self.experience))
        self.module._launch_app = repeated_factory
        with patch.object(bridge, '_CONSUMED_PATHS', set()), self.context():
            with self.assertRaises(ValueError): self.module._launch_app()
        self.factory.assert_called_once()

    def test_constructor_error_restores_module_and_noop_paths(self):
        self.factory.side_effect = RuntimeError('constructor failed')
        before = self.module.shutil, self.module.lazy, self.module._launch_app
        with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
            with self.context(): self.module._launch_app()
        self.assertEqual((self.module.shutil, self.module.lazy, self.module._launch_app), before)
        self.copy.assert_not_called()

    def test_consumed_context_cannot_relaunch_in_same_process(self):
        with self.context(): self.module._launch_app()
        self.assertIsNone(self.module.og.app)
        with self.assertRaisesRegex(RuntimeError, 'already reserved or attempted'):
            with self.context(): self.module._launch_app()
        self.factory.assert_called_once()

    def test_failed_attempt_consumes_process_gate(self):
        self.factory.side_effect = RuntimeError('native initialization failed')
        with self.assertRaisesRegex(RuntimeError, 'native initialization failed'):
            with self.context(): self.module._launch_app()
        with self.assertRaisesRegex(RuntimeError, 'already reserved or attempted'):
            with self.context(): self.module._launch_app()
        self.factory.assert_called_once()

    def test_escaped_unused_closure_is_invalid_after_context(self):
        with self.context(): escaped = self.module._launch_app
        with self.assertRaisesRegex(RuntimeError, 'one active OG launch attempt'): escaped()
        self.factory.assert_not_called()
        with self.context(): self.module._launch_app()
        self.factory.assert_called_once()

    def test_nested_context_cannot_replace_active_bridge(self):
        with self.context():
            launch = self.module._launch_app
            with self.assertRaisesRegex(RuntimeError, 'already reserved or attempted'):
                with self.context(): pass
            self.assertIs(self.module._launch_app, launch)
            launch()
        self.factory.assert_called_once()


if __name__ == '__main__': unittest.main()

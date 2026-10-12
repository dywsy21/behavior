"""No unbounded ping loop during long inference or a slow SSH/VPN interval."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

REPO=Path(__file__).resolve().parents[1]


class HeartbeatTests(unittest.TestCase):
    def test_all_custom_episode_transports_explicitly_disable_periodic_pings(self):
        for file in ('scripts/eval/full_episode_client.py','scripts/eval/champion_episode_client.py',
                     'scripts/eval/serve_causal_full_episode.py','scripts/eval/serve_champion_episode.py',
                     'src/behavior_champion/official.py'):
            calls=[node for node in ast.walk(ast.parse((REPO/file).read_text())) if isinstance(node,ast.Call)
                   and (isinstance(node.func,ast.Name) and node.func.id=='connect' or
                        isinstance(node.func,ast.Attribute) and node.func.attr=='serve')]
            self.assertEqual(len(calls),1,file)
            options={kw.arg:kw.value for kw in calls[0].keywords}
            self.assertIn('ping_interval',options,file)
            self.assertIsNone(ast.literal_eval(options['ping_interval']),file)

    @unittest.skipUnless(importlib.util.find_spec('websockets.sync'), 'Installed sync implementation required')
    def test_installed_keepalive_busy_loop_reproduction_and_disabled_start(self):
        from websockets.sync.connection import Connection
        waits=[];pings=[]
        thread=SimpleNamespace(join=waits.append,is_alive=lambda:len(pings)<100)
        fake=SimpleNamespace(ping_interval=20,ping_timeout=None,latency=25,
            recv_events_thread=thread,debug=False,ping=lambda **kwargs:pings.append(kwargs))
        # Calls the installed, unmodified loop; no sockets, real waiting or GPU.
        Connection.keepalive(fake)
        self.assertEqual(len(pings),100)
        self.assertTrue(all(wait<0 for wait in waits))
        fake.ping_interval=None
        Connection.start_keepalive(fake)
        self.assertFalse(hasattr(fake,'keepalive_thread'))

    def test_short_skill_control_and_episode_connections_disable_pings(self):
        files = {'run_short_skill_cold_worker.py': 2, 'serve_fixed_skill_fm.py': 1,
                 'serve_short_skill_rl.py': 1, 'collect_short_skill_rl.py': 1,
                 'check_skill_reward_replay.py': 1}
        for file, count in files.items():
            tree = ast.parse((REPO/'scripts/rl/memlite_online/tools'/file).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and (isinstance(node.func, ast.Name) and node.func.id == 'connect' or
                          isinstance(node.func, ast.Attribute) and node.func.attr == 'serve')]
            self.assertEqual(len(calls), count, file)
            for call in calls:
                options = {kw.arg: kw.value for kw in call.keywords}
                self.assertIn('ping_interval', options, file)
                self.assertIsNone(ast.literal_eval(options['ping_interval']), file)
                self.assertIsNone(ast.literal_eval(options['ping_timeout']), file)


if __name__=='__main__':unittest.main()

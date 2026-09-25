import json
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/vlm_sft'))
import launch_expanded_visual as launch


class ExpandedLaunchTests(unittest.TestCase):
    def test_registered_command_has_independent_group_timeout_and_no_gpu(self):
        cmd=launch.command()
        self.assertEqual(cmd[:4],['/usr/bin/timeout','--signal=TERM','--kill-after=10s','7200s'])
        self.assertEqual(cmd[-2:],['--output',str(launch.ROOT/'raw')])

    def test_post_popen_receipt_failure_reports_running_not_launch_failed(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp)/'run';original=launch.atomic_json;writes=[];stdout=StringIO()
            def write(path,value):
                writes.append(path)
                if len(writes)==2:raise OSError('PID_RECEIPT_FAILED')
                return original(path,value)
            stack.enter_context(patch.object(launch,'ROOT',root))
            stack.enter_context(patch.object(launch,'PYTHON',Path(sys.executable)))
            stack.enter_context(patch.object(launch,'code_identity',return_value='fixed'))
            stack.enter_context(patch.object(launch,'source_identity',return_value=([],{},{})))
            stack.enter_context(patch.object(launch,'atomic_json',side_effect=write))
            popen=stack.enter_context(patch.object(launch.subprocess,'Popen',return_value=SimpleNamespace(pid=123456)))
            with redirect_stdout(stdout):launch.launch()
            result=json.loads(stdout.getvalue());self.assertEqual(result['status'],'supervisor_started')
            self.assertEqual(result['supervisor_pid'],123456);self.assertIn('PID_RECEIPT_FAILED',result['launch_receipt_update_error'])
            self.assertEqual(popen.call_count,1)
            self.assertEqual(json.loads((root/'launch.json').read_text())['status'],'reserved')

    def test_success_requires_exit_zero_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(launch,'check_seal',return_value={'verified':'seal'}):
                result=launch.supervise(root,[sys.executable,'-c','import os; assert os.environ["CUDA_VISIBLE_DEVICES"]==""'],'fixed')
            self.assertEqual(result['status'],'completed');self.assertEqual(result['exit_code'],0)
            self.assertLess(result['seconds'],5)
            with self.assertRaises(ProcessLookupError):os.killpg(result['worker_group'],0)

    def test_native_hang_times_out_and_cannot_become_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);started=time.monotonic()
            cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=0.2s','0.2s',sys.executable,'-c',
                 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)']
            with self.assertRaisesRegex(RuntimeError,'failed or hit external timeout'):
                launch.supervise(root,cmd,'fixed')
            result=json.loads((root/'supervisor.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertFalse(result['training_eligible'])
            self.assertLess(time.monotonic()-started,10)
            # The native worker is killed by timeout even if the parent Python
            # collector cannot return from a decoder or thread-pool shutdown.
            self.assertNotEqual(result['exit_code'],0)

    def test_nonzero_exit_preserves_primary_despite_cleanup_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(launch,'stop_group',side_effect=RuntimeError('secondary cleanup')):
                with self.assertRaisesRegex(RuntimeError,'failed or hit external timeout: 7'):
                    launch.supervise(root,[sys.executable,'-c','raise SystemExit(7)'],'fixed')
            result=json.loads((root/'supervisor.json').read_text())
            self.assertIn('secondary cleanup',result['cleanup_error']);self.assertEqual(result['status'],'failed')

    def test_zero_exit_without_seal_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):launch.supervise(root,[sys.executable,'-c','pass'],'fixed')
            self.assertEqual(json.loads((root/'supervisor.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()

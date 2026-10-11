import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
import collect_full_episode_videos as collector


class CollectorTests(unittest.TestCase):
    def fixture(self,root):
        (root/'radio').mkdir()
        video=root/'radio/execution.mp4'
        video.write_bytes(b'fixture, not a real encoded video')
        digest=hashlib.sha256(video.read_bytes()).hexdigest()
        result={'task':'turning_on_radio','success':False}
        (root/'radio/manifest.json').write_text(json.dumps(dict(
            status='full_episode_encoded_and_decoded',output_sha256=digest,official_result=result)))
        state=dict(status='collector_failed_original_runs_untouched',completed={'radio':dict(
            video=str(video.resolve()),sha256=digest,official_result=result)},cases={'food':{'controls':16}},error='ssh 255')
        (root/'collector-status.json').write_text(json.dumps(state))
        return state

    def test_resume_keeps_original_receipt_and_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original=self.fixture(root)
            resumed=collector.prepare_output(root,True)
            self.assertEqual(resumed['completed'],original['completed'])
            self.assertNotIn('error',resumed)
            self.assertEqual(json.loads(next(root.glob('collector-status.before-resume-*')).read_text()),original)
            self.assertEqual(json.loads((root/'collector-status.json').read_text()),original)

    def test_resume_rejects_corrupt_video_and_partial_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'food').mkdir()
            with self.assertRaisesRegex(ValueError,'Partial'):
                collector.prepare_output(root,True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'radio/execution.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'differs'):
                collector.prepare_output(root,True)

    def test_cannot_resume_running_collector_or_overwrite_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.fixture(root)
            state['status']='waiting_for_original_runs'
            (root/'collector-status.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'failed'):
                collector.prepare_output(root,True)
            with self.assertRaises(FileExistsError):
                collector.prepare_output(root,False)

    def test_retries_only_read_only_transport_failures(self):
        cmd=['ssh','host','cat','status.json']
        with mock.patch.object(collector.subprocess,'check_output',side_effect=[subprocess.CalledProcessError(255,cmd),'{}']) as call, mock.patch.object(collector.time,'sleep'):
            self.assertEqual(collector.run(cmd),'{}')
            self.assertEqual(call.call_count,2)
        with mock.patch.object(collector.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,cmd)) as call, mock.patch.object(collector.time,'sleep'):
            with self.assertRaises(subprocess.CalledProcessError):
                collector.run(cmd)
            self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()

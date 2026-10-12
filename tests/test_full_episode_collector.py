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
    def test_registration_is_explicit_all_cases_and_binds_runtime_sources(self):
        source=Path(__file__).resolve().parents[1]/'configs/recovery_sft/periodic_full_video_collection_v2.json'
        spec=collector.registration(source)
        self.assertEqual(spec['pairs']['radio'],('radio-sim-v2','radio-model-v2'))
        status=dict(case='radio',config_sha256=spec['config_sha256'],source_commit=spec['sim_source_commit'])
        collector.validate_registered_status(spec,'radio',status,status)
        for field,value in [('case','food'),('config_sha256','f'*64),('source_commit','a'*40)]:
            bad=dict(status,**{field:value})
            with self.assertRaises(ValueError):collector.validate_registered_status(spec,'radio',bad,status)

    def test_registration_rejects_path_traversal_injection_duplicates_and_selection(self):
        source=Path(__file__).resolve().parents[1]/'configs/recovery_sft/periodic_full_video_collection_v2.json'
        original=json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'registration.json'
            for edit in [lambda x:x['pairs'].pop('food'),
                         lambda x:x['pairs']['radio'].__setitem__(0,'../../foreign'),
                         lambda x:x.__setitem__('rtx_root',x['rtx_root']+'; bad'),
                         lambda x:x['pairs']['food'].__setitem__(0,x['pairs']['radio'][0])]:
                value=json.loads(json.dumps(original));edit(value);path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):collector.registration(path)

    def test_resume_refuses_to_change_registration_even_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';state=collector.prepare_output(root,False,registration_sha256='a'*64)
            state['status']='collector_failed_original_runs_untouched'
            (root/'collector-status.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'Registration'):
                collector.prepare_output(root,True,registration_sha256='b'*64)
            self.assertFalse(list(root.glob('collector-status.before-resume-*')))

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

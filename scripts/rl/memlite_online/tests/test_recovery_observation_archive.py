from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from skill_observation_archive import SkillObservationArchive,load_observation,audit_archive
from skill_aligned_reward import SkillIdentity
from test_recovery_skill_training_protocol import observation


class ObservationArchiveTests(unittest.TestCase):
    def test_lossless_two_chunks_do_not_mix_images_clock_or_policy(self):
        identity=SkillIdentity('run','task',1,'episode','ctx','a'*64,0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'obs';archive=SkillObservationArchive(root,identity,policy_version=7,policy_sha256='b'*64,start_control=32)
            obs=observation();rng=np.random.default_rng(17)
            for x in obs['images'].values():x[:]=rng.integers(0,256,size=x.shape,dtype=np.uint8)
            obs['proprio'][:]=rng.normal(size=61).astype(np.float32)
            first=archive.append(identity,32,obs);obs['images']['head_rgb'][0,0,0]^=255
            second=archive.append(identity,48,obs)
            a=load_observation(root,first);b=load_observation(root,second)
            self.assertNotEqual(a['images']['head_rgb'][0,0,0],b['images']['head_rgb'][0,0,0])
            for name,x in obs['images'].items():np.testing.assert_array_equal(b['images'][name],x)
            np.testing.assert_array_equal(b['proprio'],obs['proprio'])
            header=json.loads((root/'manifest.json').read_text())
            self.assertEqual(header['policy_version'],7);self.assertFalse(header['admission_for_training'])
            accepted=audit_archive(root,identity,policy_version=7,policy_sha256='b'*64,expected_steps=[32,48])
            self.assertEqual(accepted['observations'],2)
            with self.assertRaises(ValueError):audit_archive(root,identity,policy_version=7,policy_sha256='b'*64,expected_steps=[32,40,48])
            with self.assertRaises(ValueError):audit_archive(root,identity,policy_version=8,policy_sha256='b'*64,expected_steps=[32,48])
            for ident,step in [(identity,48),(identity,16),(replace(identity,task='other'),64)]:
                with self.assertRaises(ValueError):archive.append(ident,step,obs)
            with self.assertRaises(FileExistsError):SkillObservationArchive(root,identity,policy_version=8,policy_sha256='c'*64,start_control=0)
            bad=dict(second,observation_sha256='d'*64)
            with self.assertRaises(ValueError):load_observation(root,bad)
            bad=dict(second,file='../outside.npz')
            with self.assertRaises(ValueError):load_observation(root,bad)

    def test_rejects_privileged_fields_or_skipped_initial_observation(self):
        identity=SkillIdentity('run','task',1,'episode','ctx','a'*64,0)
        with tempfile.TemporaryDirectory() as tmp:
            archive=SkillObservationArchive(Path(tmp)/'obs',identity,policy_version=0,policy_sha256='b'*64,start_control=32)
            with self.assertRaises(ValueError):archive.append(identity,48,observation())
            with self.assertRaises(ValueError):archive.append(identity,32,dict(observation(),success=True))
            bad=observation();bad['proprio']=bad['proprio'].astype(np.float64)
            with self.assertRaises(ValueError):archive.append(identity,32,bad)


if __name__=='__main__':unittest.main()

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
from native_rl_profile import traced_imports,session


@dataclass
class Imports:
    Evaluator: object
    OmegaConf: object


class ProfileTests(unittest.TestCase):
    def test_public_sessions_require_explicit_evaluation_role(self):
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(ValueError,'split'):
                with session(None,SimpleNamespace(official_mode='public_test'),gpu=2,output=None,runtime=None):
                    pass
            with self.assertRaisesRegex(ValueError,'split'):
                with session(None,SimpleNamespace(official_mode='train'),gpu=2,output=None,runtime=None,
                             evaluation_only=True):
                    pass

    def test_fixed_instance_multiple_official_resets_and_event_failure(self):
        class Env:
            env='robot'
            resets=0
            def reset(self): self.resets+=1; return self.resets
            def load_task_instance(self,i): return i
        record={}
        imports=Imports(lambda cfg:Env(),SimpleNamespace(to_container=lambda cfg,resolve:cfg,create=lambda cfg:cfg))
        with patch('shared_camera_config.prepare_config',return_value=({},dict(cameras=[]))),patch('shared_camera_config.validate_wrapper'):
            traced=traced_imports(imports,record,instance_id=138,write=lambda _:None)
            env=traced.Evaluator({})
            self.assertEqual(env.reset(),1)
            self.assertEqual(env.load_task_instance(138),138)
            for expected in range(2,7): self.assertEqual(env.reset(),expected)
            with self.assertRaises(ValueError): env.load_task_instance(301)
            self.assertEqual((record['reset_count'],record['load_count']),(6,1))
            self.assertTrue(all(e['status']=='completed' for e in record['events']))
            with self.assertRaises(ValueError): traced.Evaluator({})

    def test_unfinished_original_reset_blocks_next_event(self):
        class Env:
            env='robot'
            def reset(self): raise RuntimeError('native reset failed')
        imports=Imports(lambda cfg:Env(),SimpleNamespace(to_container=lambda cfg,resolve:cfg,create=lambda cfg:cfg))
        record={}
        with patch('shared_camera_config.prepare_config',return_value=({},dict(cameras=[]))),patch('shared_camera_config.validate_wrapper'):
            env=traced_imports(imports,record,instance_id=1,write=lambda _:None).Evaluator({})
            with self.assertRaises(RuntimeError): env.reset()
            with self.assertRaises(RuntimeError): env.reset()
            self.assertEqual(record['reset_count'],0)


if __name__=='__main__': unittest.main()

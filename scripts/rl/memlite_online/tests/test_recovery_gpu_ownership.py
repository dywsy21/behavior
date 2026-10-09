import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_gpu_ownership import owns_auxiliary


class OwnershipTests(unittest.TestCase):
    def test_only_same_registered_peer_small_aux_context_can_share(self):
        args=(123,188,7,dict(pid=123),'/tmp/owned/case',
              ['python','collect_local_grasp_recovery.py','--output','/tmp/owned/case'],dict(EVAL_GPU='2'))
        self.assertTrue(owns_auxiliary(*args))
        for i,replacement in [(0,124),(1,2000),(1,float('nan')),(2,2),(3,dict(pid=124)),(4,'/tmp/other'),(6,{})]:
            changed=list(args);changed[i]=replacement;self.assertFalse(owns_auxiliary(*changed))


if __name__=='__main__':unittest.main()

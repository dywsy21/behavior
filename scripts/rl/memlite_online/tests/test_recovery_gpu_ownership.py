import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_gpu_ownership import owns_auxiliary,owns_short_skill_auxiliary


class OwnershipTests(unittest.TestCase):
    def test_short_skill_aux_requires_exact_recipe_port_case_and_other_gpu(self):
        args=(123,174,2,dict(pid=123,config_sha256='abc',case='radio'),'/tmp/run/radio',
              ['python','collect_short_skill_rl.py','--output','/tmp/run/radio','--port','18973'],dict(EVAL_GPU='0'))
        kwargs=dict(config_sha256='abc',cases={'radio'},port=18973)
        self.assertTrue(owns_short_skill_auxiliary(*args,**kwargs))
        for key,value in [('config_sha256','other'),('cases',{'wash'}),('port',1)]:
            self.assertFalse(owns_short_skill_auxiliary(*args,**(kwargs|{key:value})))
        changed=list(args);changed[1]=15000
        self.assertFalse(owns_short_skill_auxiliary(*changed,**kwargs))
        self.assertFalse(owns_auxiliary(*args))

    def test_only_same_registered_peer_small_aux_context_can_share(self):
        args=(123,188,7,dict(pid=123),'/tmp/owned/case',
              ['python','collect_local_grasp_recovery.py','--output','/tmp/owned/case'],dict(EVAL_GPU='2'))
        self.assertTrue(owns_auxiliary(*args))
        placement=list(args);placement[5]=['python','collect_placement_curriculum.py','--output','/tmp/owned/case']
        self.assertTrue(owns_auxiliary(*placement))
        wrong=list(args);wrong[5]=['python','unrelated_training.py','--output','/tmp/owned/case']
        self.assertFalse(owns_auxiliary(*wrong))
        for i,replacement in [(0,124),(1,2000),(1,float('nan')),(2,2),(3,dict(pid=124)),(4,'/tmp/other'),(6,{})]:
            changed=list(args);changed[i]=replacement;self.assertFalse(owns_auxiliary(*changed))


if __name__=='__main__':unittest.main()

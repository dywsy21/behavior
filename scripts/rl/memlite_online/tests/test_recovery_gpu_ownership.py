import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_gpu_ownership import owns_auxiliary,owns_short_skill_auxiliary,owns_collection_auxiliary,declared_collection_peers


class OwnershipTests(unittest.TestCase):
    def test_multiple_predeclared_waves_are_not_blanket_process_permissions(self):
        peers=[dict(collection='/tmp/owned/cal90',source_commit='a'*40),
               dict(collection='/tmp/owned/cal97',source_commit='b'*40)]
        self.assertEqual(declared_collection_peers(peers),peers)
        receipt=dict(schema='local_recovery_collection_status_v1',pid=123,source_commit='b'*40)
        args=[123,174,0,receipt,'/tmp/owned/cal97/case',
            ['python','collect_local_grasp_recovery.py','--output','/tmp/owned/cal97/case'],dict(EVAL_GPU='4')]
        self.assertFalse(owns_collection_auxiliary(*args,**peers[0]))
        self.assertTrue(owns_collection_auxiliary(*args,**peers[1]))
        args[1]=513
        self.assertFalse(any(owns_collection_auxiliary(*args,**p) for p in peers))
        for name in ('/','/tmp','relative/name','/tmp/owned/*','/tmp/owned/../other'):
            with self.assertRaises(ValueError):declared_collection_peers([dict(collection=name,source_commit='a'*40)])
        with self.assertRaises(ValueError):declared_collection_peers(peers+peers[:1])
        with self.assertRaises(ValueError):declared_collection_peers([],legacy_path='/tmp/owned/cal90')
        self.assertEqual(declared_collection_peers([],legacy_path='/tmp/owned/cal90',legacy_commit='a'*40),peers[:1])

    def test_collection_peer_requires_exact_frozen_source_parent_and_pid(self):
        receipt=dict(schema='local_recovery_collection_status_v1',pid=123,source_commit='a'*40)
        args=[123,174,0,receipt,'/tmp/owned/calibration/case',
              ['python','collect_local_grasp_recovery.py','--output','/tmp/owned/calibration/case'],dict(EVAL_GPU='4')]
        binding=dict(collection='/tmp/owned/calibration',source_commit='a'*40)
        self.assertTrue(owns_collection_auxiliary(*args,**binding))
        for key,value in [('collection','/tmp/owned'),('source_commit','b'*40),('source_commit','bad')]:
            self.assertFalse(owns_collection_auxiliary(*args,**(binding|{key:value})))
        for index,value in [(1,12000),(2,4),(4,'/tmp/owned/calibration/sibling'),
                            (3,dict(receipt,pid=124)),(3,dict(receipt,schema='unowned'))]:
            changed=list(args);changed[index]=value
            self.assertFalse(owns_collection_auxiliary(*changed,**binding))

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

    def test_short_skill_peer_never_authorizes_active_primary_or_foreign_output(self):
        args=[123,180,5,dict(pid=123,config_sha256='abc',case='radio'),
              '/tmp/owned/radio-episode-000003',
              ['python','collect_short_skill_rl.py','--output','/tmp/owned/radio-episode-000003','--port','18975'],
              dict(EVAL_GPU='0')]
        binding=dict(config_sha256='abc',cases={'radio'},port=18975)
        self.assertTrue(owns_short_skill_auxiliary(*args,**binding))
        for index,value in [(2,0),(1,12000),(4,'/tmp/teammate/run'),(6,dict(EVAL_GPU='5')),
                            (3,dict(pid=124,config_sha256='abc',case='radio'))]:
            changed=list(args);changed[index]=value
            self.assertFalse(owns_short_skill_auxiliary(*changed,**binding))

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

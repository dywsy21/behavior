from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from skill_cold_worker import next_worker_action,validate_baseline_acceptance,finished_workers


class ColdWorkerTests(unittest.TestCase):
    def test_only_one_completed_episode_or_explicit_service_finish(self):
        result=dict(status='completed_single_episode',completed_episodes=1,fresh_process_per_episode=True)
        self.assertEqual(next_worker_action(0,result),'next_episode')
        self.assertEqual(next_worker_action(0,dict(status='server_finished',completed_episodes=0)),'finished')
        for changed in (dict(result,completed_episodes=2),dict(result,fresh_process_per_episode=False),
                        dict(result,status='failed'),dict(result,status='server_closed_between_episodes'),None):
            with self.assertRaises(ValueError):next_worker_action(0,changed)
        with self.assertRaises(ValueError):next_worker_action(1,result)

    def test_all_workers_receive_explicit_finish(self):
        seen=set();cases=['grasp','open','place']
        self.assertFalse(finished_workers(cases,seen,'grasp'))
        self.assertFalse(finished_workers(cases,seen,'grasp'))
        with self.assertRaises(ValueError):finished_workers(cases,seen,'foreign')
        self.assertFalse(finished_workers(cases,seen,'place'))
        self.assertTrue(finished_workers(cases,seen,'open'))

    def test_baseline_acceptance_cannot_cross_run_or_checkpoint(self):
        binding=dict(config_sha256='c'*64,policy_sha256='p'*64,episodes_sha256='e'*64)
        receipt=dict(schema='short_skill_cold_baseline_acceptance_v1',approved=True,
                     reviewed_episodes=6,**binding)
        validate_baseline_acceptance(receipt,**binding)
        for key in binding:
            with self.assertRaises(ValueError):
                validate_baseline_acceptance(dict(receipt,**{key:'x'*64}),**binding)
        with self.assertRaises(ValueError):validate_baseline_acceptance(dict(receipt,approved=False),**binding)


if __name__=='__main__':unittest.main()

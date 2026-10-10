from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from causal_skill_probe_audit import audit_joint_episode
from skill_training_protocol import observation_hash
import test_recovery_causal_skill_probe as probe_test


class AuditTests(unittest.TestCase):
    def setUp(self):
        t=probe_test.ProbeTests();t.setUp();p,j,_=t.start()
        controls=t.apply(p,j,16);t.probe.goal(p,j,t.observation);controls+=t.apply(p,j,3)
        p.rollout.ended=True;t.probe.close(p,j)
        self.events=[dict(kind=k,**v) for k,v in t.log]
        self.kwargs=dict(models=t.models,history_row=t.manifest['rows'][0],case=p.case,job=j,
            controls=controls,observation_hashes={k:observation_hash(t.observation) for k in (32,48,51)})

    def test_independent_replay_matches_actual_partial_controls_and_intents(self):
        result=audit_joint_episode(self.events,**self.kwargs)
        self.assertEqual(result['applied_controls'],19);self.assertEqual(result['chunks'],2)
        self.assertEqual(result['generations'],1);self.assertEqual(result['reuses'],1)
        self.assertEqual(result['final_control_step'],51)
        self.assertTrue(result['all_current_rgb_proprio_inputs_bound'])

    def test_wrong_current_rgb_history_cadence_or_goal_rejected(self):
        for mutate in (lambda e:e[1].update(observation_sha256='bad'),
                       lambda e:e[1]['result']['causal_input'].update(memory='foreign'),
                       lambda e:e[4]['result'].update(reused=False),
                       lambda e:e[2]['goal'].update(parent_goal='different')):
            bad=deepcopy(self.events);mutate(bad)
            with self.assertRaises(ValueError):audit_joint_episode(bad,**self.kwargs)

    def test_changed_offered_or_actual_action_or_wrong_ack_clock_rejected(self):
        bad=deepcopy(self.events);bad[2]['offered_actions_raw23'][0][0]=.25
        with self.assertRaises(ValueError):audit_joint_episode(bad,**self.kwargs)
        for mutate in (lambda x:x[0]['action_executed_raw23'].__setitem__(0,.5),
                       lambda x:x[0].update(control_step=34),lambda x:x[0].update(simulator_apply_ack=False)):
            kwargs=deepcopy(self.kwargs);mutate(kwargs['controls'])
            with self.assertRaises(ValueError):audit_joint_episode(self.events,**kwargs)

    def test_missing_duplicate_foreign_or_invented_truth_rejected(self):
        for mutate in (lambda e:e.pop(3),lambda e:e.insert(2,deepcopy(e[1])),
                       lambda e:e[2]['job'].update(id='other'),
                       lambda e:e[1]['result'].update(physical_success_asserted=True),
                       lambda e:e[-1].update(control_step=52)):
            bad=deepcopy(self.events);mutate(bad)
            with self.assertRaises(ValueError):audit_joint_episode(bad,**self.kwargs)


if __name__=='__main__':unittest.main()

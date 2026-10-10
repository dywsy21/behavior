import json
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import digest,group_key,split_group
from skill_rounds import SkillRounds
from skill_training_protocol import read_only_recipe


class ShortSkillRecipeTests(unittest.TestCase):
    def test_final_transport_ack_does_not_need_an_extra_physical_reset(self):
        from skill_cold_worker import probe_finish_without_simulator,require_probe_finish_response
        for completed in range(8):self.assertFalse(probe_finish_without_simulator(completed,list(range(8))))
        self.assertTrue(probe_finish_without_simulator(8,list(range(8))))
        for completed,seeds in [(9,list(range(8))),(-1,[17]),(True,[17]),(1,[]),(2,[17,17])]:
            with self.assertRaises(ValueError):probe_finish_without_simulator(completed,seeds)
        self.assertFalse(require_probe_finish_response({'status':'wait'}))
        self.assertTrue(require_probe_finish_response({'status':'finished'}))
        for response in ({'status':'job'},{'status':'finished','job':{}},{'error':'failed'},None):
            with self.assertRaises(ValueError):require_probe_finish_response(response)

    def test_fresh_noise_replication_freezes_both_final_policies_and_conditions(self):
        from copy import deepcopy
        replicas=[]
        for arm in ('control','later_actions'):
            old=json.loads((REPO/f'configs/recovery_sft/a800_later_action_probes_{arm}_v1.json').read_text())
            new=json.loads((REPO/f'configs/recovery_sft/a800_later_action_replication_{arm}_v1.json').read_text())
            self.assertTrue(read_only_recipe(new))
            for key in ('cases','model','stats_sha256','ppo','seed','observation_archive'):
                self.assertEqual(old[key],new[key])
            self.assertEqual(len(new['evaluation_seeds']),8)
            self.assertEqual(len(set(new['evaluation_seeds'])),8)
            self.assertFalse(set(new['evaluation_seeds']) & set(old['evaluation_seeds']))
            self.assertFalse(new['replication']['combine_with_prior_for_primary_claim'])
            self.assertFalse(new['replication']['task_generalization'])
            self.assertEqual(new['pair']['seed_roles'].keys(),set(map(str,new['evaluation_seeds'])))
            normalized=deepcopy(new);del normalized['model'];del normalized['pair']['arm']
            replicas.append(normalized)
        self.assertEqual(*replicas)

    def test_hello_readiness_requires_exact_protocol_and_binding(self):
        from skill_cold_worker import require_service_hello
        sha='a'*64
        self.assertTrue(require_service_hello(dict(protocol='short_skill_rl_v1',config_sha256=sha),sha))
        for value in (dict(protocol='other',config_sha256=sha),
                      dict(protocol='short_skill_rl_v1',config_sha256='b'*64),
                      dict(protocol='short_skill_rl_v1',config_sha256=sha,job='already-claimed'),
                      {'status':'ready'},None):
            with self.assertRaises(ValueError):require_service_hello(value,sha)

    def test_both_pinned_sft_probes_use_shared_readonly_mode_gate(self):
        for name in ('control','later_actions'):
            cfg=json.loads((REPO/f'configs/recovery_sft/a800_later_action_probes_{name}_v1.json').read_text())
            self.assertTrue(read_only_recipe(cfg))
            for change in (dict(schema='unregistered'),dict(learning_rounds=1),dict(learning_rounds=False),
                    dict(resume={'path':'old-ppo'}),dict(training_rollout_multiplier=4),
                    dict(simulator_reset_protocol='warm'),dict(user_goal_authorized=False)):
                with self.assertRaises(ValueError):read_only_recipe(dict(cfg,**change))
        cfg=json.loads((REPO/'configs/recovery_sft/a800_short_skill_rl_continuation_v7.json').read_text())
        self.assertFalse(read_only_recipe(cfg))

    def test_three_train_mechanisms_with_bound_proofs(self):
        recipe=json.loads((REPO/'configs/recovery_sft/a800_short_skill_rl_v1.json').read_text())
        self.assertTrue(recipe['user_goal_authorized'])
        verbs=set()
        for key,case in recipe['cases'].items():
            self.assertEqual(key,case['task']+'_'+str(case['instance_id']))
            self.assertEqual(case['original_split'],'train')
            self.assertEqual(case['recovery_split'],split_group(case['task'],case['instance_id']))
            self.assertEqual(case['recovery_split'],'train')
            self.assertLess(case['start_control'],case['end_control'])
            for sha in (case['context_id'],case['manifest_sha256'],case['sim']['proof_sha256']):
                self.assertEqual(len(sha),64);self.assertEqual(set(sha)-set('0123456789abcdef'),set())
            bundle=json.loads(case['semantic_bundle']);self.assertEqual(len(bundle),1);verbs.add(bundle[0]['verb'])
            if case['sim']['kind']=='placement':
                self.assertEqual(case['context_id'],digest([group_key(case['task'],case['instance_id']),
                    case['start_control']-1,case['semantic_bundle']]))
        self.assertEqual(verbs,{'GRASP','OPEN_DOOR','PLACE_ON'})
        rounds=SkillRounds(recipe['cases'],recipe['evaluation_seeds'],rounds=recipe['learning_rounds'],run='unit')
        self.assertEqual(len(rounds.jobs),6)
        self.assertTrue(all(j['phase']=='evaluation' for j in rounds.jobs))
        self.assertLessEqual(recipe['ppo']['target_kl'],.05)

    def test_critic_fix_does_not_expand_actor_step_or_reuse_incomplete_seed(self):
        before=json.loads((REPO/'configs/recovery_sft/a800_short_skill_rl_cold_resume_v2.json').read_text())
        after=json.loads((REPO/'configs/recovery_sft/a800_short_skill_rl_cold_resume_v3.json').read_text())
        self.assertEqual(before['cases'],after['cases']);self.assertEqual(before['model'],after['model'])
        self.assertTrue(after['require_baseline_acceptance'])
        self.assertEqual(after['resume']['updates'],12)
        self.assertEqual(after['training_seed_round_offset'],13)
        self.assertEqual(after['learning_rounds']+after['resume']['updates'],20)
        self.assertEqual(after['observation_archive'],'lossless_chunk_boundaries_v1')
        self.assertTrue(after['ppo']['critic_restart_stale_momentum']);self.assertTrue(after['ppo']['critic_update_audit'])
        for key,value in before['ppo'].items():self.assertEqual(after['ppo'][key],value)

    def test_continuation_only_changes_train_horizon_and_probe_cadence(self):
        from recovery_gpu_ownership import declared_collection_peers
        from skill_training_protocol import rollout_end_control
        before=json.loads((REPO/'configs/recovery_sft/a800_short_skill_rl_cold_resume_v3.json').read_text())
        after=json.loads((REPO/'configs/recovery_sft/a800_short_skill_rl_continuation_v7.json').read_text())
        self.assertEqual(before['cases'],after['cases']);self.assertEqual(before['model'],after['model'])
        self.assertEqual(after['resume']['updates'],20)
        self.assertEqual(after['training_seed_round_offset'],21)
        self.assertTrue(after['require_baseline_acceptance'])
        self.assertEqual(after['evaluation_seeds'],before['evaluation_seeds'])
        self.assertEqual(after['evaluation_interval_updates'],4)
        self.assertEqual(len(declared_collection_peers(after['collection_peers'])),2)
        self.assertEqual(after['ppo']['critic_fit_steps'],1)
        for key,value in before['ppo'].items():self.assertEqual(after['ppo'][key],value)
        for case in after['cases'].values():
            self.assertEqual(rollout_end_control(case,'evaluation',4),case['end_control'])
            self.assertEqual(rollout_end_control(case,'train',4)-case['start_control'],
                4*(case['end_control']-case['start_control']))


if __name__=='__main__':unittest.main()

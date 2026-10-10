import json
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import digest,group_key,split_group
from skill_rounds import SkillRounds


class ShortSkillRecipeTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()

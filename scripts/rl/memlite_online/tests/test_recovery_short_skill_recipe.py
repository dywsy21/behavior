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


if __name__=='__main__':unittest.main()

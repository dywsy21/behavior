from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from skill_rounds import SkillRounds


class RoundTests(unittest.TestCase):
    def make(self):return SkillRounds(['grasp','open','place'],[17,29],rounds=2,run='r')

    def complete(self,r):
        index=0
        for case in r.cases:
            while (job:=r.take(case)) is not None:
                r.complete(job['id'],[dict(experience_id=index)] if r.phase=='train' else [])
                index+=1

    def test_no_cross_version_update_during_active_episode(self):
        r=self.make();r.take('open')
        with self.assertRaises(ValueError):r.advance()
        self.assertFalse(r.ready)

    def test_paired_probes_never_enter_training_and_versions_are_barriered(self):
        r=self.make();self.assertEqual(len(r.jobs),6);self.complete(r);r.advance()
        self.assertEqual((r.phase,r.round,len(r.jobs)),('train',1,3))
        self.complete(r)
        with self.assertRaises(ValueError):r.advance()
        r.advance(optimizer_completed=True);self.assertEqual((r.phase,r.round),('evaluation',1))
        self.complete(r);r.advance();self.complete(r);r.advance(optimizer_completed=True)
        self.complete(r);r.advance();self.assertEqual((r.phase,r.round),('finished',2))

    def test_duplicate_completion_or_eval_targets_rejected(self):
        r=self.make();job=r.take('open')
        with self.assertRaises(ValueError):r.complete(job['id'],[dict(experience_id=0)])
        r.complete(job['id'],[])
        with self.assertRaises(ValueError):r.complete(job['id'],[])
        with self.assertRaises(ValueError):r.take('unknown')

    def test_reused_training_experience_not_allowed(self):
        r=self.make();self.complete(r);r.advance()
        a=r.take('open');b=r.take('grasp')
        r.complete(a['id'],[dict(experience_id=0)])
        with self.assertRaises(ValueError):r.complete(b['id'],[dict(experience_id=0)])

    def test_resume_uses_fresh_training_seeds_and_unchanged_paired_probes(self):
        r=SkillRounds(['grasp','open','place'],[17,29],rounds=13,run='new',seed_round_offset=7)
        self.assertEqual({j['seed'] for j in r.jobs},{17,29})
        self.complete(r);r.advance()
        self.assertEqual({j['seed'] for j in r.jobs},{25072})
        self.complete(r);r.advance(optimizer_completed=True)
        self.assertEqual({j['seed'] for j in r.jobs},{17,29})

    def test_invalid_or_overlapping_seed_offset_is_rejected(self):
        for offset in (-1,True,0.5):
            with self.assertRaises(ValueError):
                SkillRounds(['grasp','open','place'],[17],rounds=1,run='r',seed_round_offset=offset)
        with self.assertRaises(ValueError):
            SkillRounds(['grasp','open','place'],[25072],rounds=1,run='r',seed_round_offset=7)


if __name__=='__main__':unittest.main()

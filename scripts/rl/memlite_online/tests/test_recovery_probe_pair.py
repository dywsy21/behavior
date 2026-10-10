from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'scripts/rl/memlite_online/code'))
from recovery_probe_pair import compare_probe_pair


class PairTests(unittest.TestCase):
    def setUp(self):
        self.configs = [json.loads((REPO/f'configs/recovery_sft/a800_later_action_probes_{arm}_v1.json').read_text())
                        for arm in ('control', 'later_actions')]
        self.services, self.audits = [], []
        self.shas = ['a'*64, 'b'*64]
        for cfg, sha in zip(self.configs, self.shas):
            jobs = [dict(case=c, seed=s, status='done', phase='evaluation', round=0)
                    for c in cfg['cases'] for s in cfg['evaluation_seeds']]
            self.services.append(dict(status='completed_read_only_skill_probes', error=None,
                evaluation_only=True, optimizer_updates=0, actor_updates=0, active_episodes=0,
                whole_task_sr_evaluated=False, config_sha256=sha, parent_sha256=cfg['model']['sha256'],
                policy_identity_sha256=cfg['model']['sha256'], actor_before_sha256='c'*64,
                actor_after_sha256='c'*64, frozen_before_sha256='d'*64, frozen_after_sha256='d'*64,
                finish_acknowledged=list(cfg['cases']), jobs=jobs))
            self.audits.append(dict(status='machine_control_reward_reset_audit_passed', baseline_protocol_checked=True,
                optimizer_authorized=False, whole_task_sr=False, config_sha256=sha, policy_sha256=cfg['model']['sha256'],
                episodes=[dict(case=j['case'], seed=j['seed'], success=True, outcome='SUCCEEDED', actual_controls=20,
                               reset_proprio_error=0., reset_target_error=0., observation_archive={}) for j in jobs]))

    def run_pair(self):
        return compare_probe_pair(self.configs, self.services, self.audits, self.shas)

    def test_complete_pair_and_conditional_control_difference(self):
        row = self.audits[1]['episodes'][0]
        row.update(success=False, outcome='UNKNOWN', actual_controls=141)
        value = self.run_pair()
        self.assertEqual(len(value['pairs']), 12)
        self.assertFalse(value['policy_promotion_authorized'])
        self.assertFalse(value['whole_task_sr'])
        lost = [p for p in value['pairs'] if p['outcome_delta'] == -1]
        self.assertEqual(len(lost), 1)
        self.assertIsNone(lost[0]['controls_delta_if_both_success'])

    def test_reject_unpaired_scientific_conditions(self):
        original = deepcopy(self.configs)
        for change in (dict(seed=1), dict(evaluation_seeds=[17, 29]), dict(stats_sha256='x')):
            self.configs = deepcopy(original); self.configs[1].update(change)
            with self.assertRaises(ValueError): self.run_pair()

    def test_reject_incomplete_service_and_updates(self):
        original = deepcopy(self.services)
        for change in (dict(optimizer_updates=1), dict(actor_after_sha256='x'), dict(frozen_after_sha256='x'),
                       dict(finish_acknowledged=[]), dict(active_episodes=1), dict(error='stopped'),
                       dict(policy_identity_sha256='wrong')):
            self.services = deepcopy(original); self.services[0].update(change)
            with self.assertRaises(ValueError): self.run_pair()

    def test_reject_duplicate_missing_or_foreign_episodes(self):
        original = deepcopy(self.audits)
        for transform in (lambda x: x.pop(), lambda x: x.append(deepcopy(x[0])),
                          lambda x: x[0].update(seed=999)):
            self.audits = deepcopy(original); transform(self.audits[1]['episodes'])
            with self.assertRaises(ValueError): self.run_pair()

    def test_reject_diagnostic_or_bad_outcome_clock_restore(self):
        original = deepcopy(self.audits)
        for change in (dict(actual_controls=10000), dict(outcome='UNKNOWN'), dict(reset_proprio_error=float('nan')),
                       dict(reset_target_error=.1), dict(success=1)):
            self.audits = deepcopy(original); self.audits[0]['episodes'][0].update(change)
            with self.assertRaises(ValueError): self.run_pair()
        self.audits = deepcopy(original); self.audits[0]['baseline_protocol_checked'] = False
        with self.assertRaises(ValueError): self.run_pair()


if __name__ == '__main__': unittest.main()

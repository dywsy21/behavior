import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_feedback import requests_for_candidate
import test_recovery_periodic_teacher as fixture


class HandoffTests(unittest.TestCase):
    def example(self):
        helper = fixture.PeriodicTeacherTests(); row, episode = helper.candidate()
        module = fixture.module
        labels = module.original_label_module()
        anchors = labels.anchor_records(episode['segments'], 0)
        records, audit = module.periodic_annotation_trace(anchors, episode['segments'])
        rec = records[2]; segment = episode['segments'][rec['segment']]
        row['projection'] = labels.projection(segment, branch='high', task_name=episode['task_name'],
            previous_intent=rec['previous_intent'], previous_parent=rec['previous_parent'], history=rec['history'])
        row['projection'].update(execution_feedback=module.unknown_feedback(2, records, episode['segments']),
                                 low_action_supervision_mask=False)
        row['identity'].update(frame=256, original_candidate_index=108)
        row['original_stride16_previous_intent'] = anchors[16][2]
        row['annotation_trace'] = audit
        return row, {frame:108+(frame-256)//16 for frame in range(192,257,16)}

    def test_two_four_frame_checks_use_previous_intent_not_next_label(self):
        row, frames = self.example()
        out, reason = requests_for_candidate(row, frames, excluded_groups=set())
        self.assertIsNone(reason)
        self.assertEqual(out['target_change_class'], 'explicit_verb_or_bundle_change')
        self.assertEqual([[c['frame'] for c in r['checks']] for r in out['requests']],
                         [[192,208,224,240], [208,224,240,256]])
        for r in out['requests']:
            for c in r['checks']:
                self.assertEqual(json.loads(c['issued_bundle'])[0]['verb'], 'GRASP')
                self.assertEqual(c['parent_goal'], row['projection']['previous_parent_goal'])
                self.assertEqual(c['memory'], row['projection']['memory'])
                self.assertEqual(c['served_controls'], c['frame']-128)
                self.assertNotIn('active_skills_text', c)
        self.assertIn('PLACE_ON', out['original_projection']['active_skills_text'])
        self.assertFalse(out['physical_outcome_asserted'])

    def test_source_exclusions_and_missing_observation_never_fill_in(self):
        row, frames = self.example()
        self.assertEqual(requests_for_candidate(row, frames, excluded_groups={'vacuuming floors:1'}),
                         (None, 'protected_source'))
        del frames[208]
        self.assertEqual(requests_for_candidate(row, frames, excluded_groups=set()),
                         (None, 'missing_original_annotated_observation'))

    def test_eval_future_or_outcome_labels_rejected(self):
        row, frames = self.example()
        for section, field, value in [('identity','split','eval'),('identity','instance',301),
                                      ('identity','frame',272),('projection','outcome_supervision_mask',True),
                                      ('projection','known_previous_outcome','SUCCEEDED')]:
            changed = copy.deepcopy(row); changed[section][field] = value
            with self.assertRaises(ValueError):
                requests_for_candidate(changed, frames, excluded_groups=set())

    def test_two_fresh_checks_equal_complete_ledger_not_stale_success_latch(self):
        from g05.utils.memlite_causal_feedback import CausalFeedbackLedger, FeedbackIdentity
        from g05.utils.memlite_skill_protocol import canonical_json, parse_active_skills_text
        row, _ = self.example()
        identity = FeedbackIdentity('audit', 'vacuuming floors', 1, 'original_train', 'a'*64, 'b'*64)
        bundle = canonical_json(parse_active_skills_text(row['projection']['previous_intent']))
        for last in ('SUCCEEDED', 'FAILED', 'UNKNOWN', 'IN_PROGRESS'):
            for before in ('SUCCEEDED', 'FAILED', 'UNKNOWN', 'IN_PROGRESS'):
                ledgers = [CausalFeedbackLedger(identity), CausalFeedbackLedger(identity)]
                for t in range(0,513,16):
                    for number, ledger in enumerate(ledgers):
                        ledger.observe_clock(identity,t)
                        if t < 512 and t % 128 == 0:
                            ledger.issued(identity,t,bundle,'Task goal: vacuuming floors')
                        if number == 0 or t >= 496:
                            value = last if t == 512 else before if t == 496 else 'SUCCEEDED'
                            ledger.estimated(identity,t,[dict(outcome=value,confidence=.96)],calibrated=True)
                self.assertEqual(ledgers[0].projection(identity,512), ledgers[1].projection(identity,512))


if __name__ == '__main__': unittest.main()

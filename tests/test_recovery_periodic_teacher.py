import importlib.util
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/rl/memlite_online/code'))
spec = importlib.util.spec_from_file_location('periodic_teacher',
    Path(__file__).resolve().parents[1] / 'scripts/rl/memlite_online/code/recovery_periodic_teacher.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PeriodicTeacherTests(unittest.TestCase):
    def candidate(self):
        from g05.utils.memlite_skill_protocol import parse_active_skills_text, canonical_json
        labels = module.original_label_module()
        anchor_records, projection = labels.anchor_records, labels.projection
        segments = []
        for start, end, verb in ((0,64,'NAVIGATE'),(64,192,'GRASP'),(192,512,'PLACE_ON')):
            text = self.command(verb=verb)
            segments.append(dict(start=start,end=end,text=text,semantic=canonical_json(parse_active_skills_text(text)),
                                 parent='Task goal: vacuuming floors',parent_supervised=False))
        episode = dict(row=dict(episode_index=7,task_index=0,task_instance_id=1,raw_episode_id=10),
                       segments=segments,phase=0,split='train',task_name='vacuuming floors')
        anchors=anchor_records(segments,0); records,audit=module.periodic_annotation_trace(anchors,segments)
        record=records[1]; segment=segments[record['segment']]
        target=projection(segment,branch='high',task_name=episode['task_name'],previous_intent=record['previous_intent'],
                          previous_parent=record['previous_parent'],history=record['history'])
        target.update(execution_feedback=module.unknown_feedback(1,records,segments),low_action_supervision_mask=False)
        identity=dict(episode_serial=7,original_candidate_index=100,episode=7,task=0,instance=1,raw_episode=10,
                      frame=128,phase=0,split='train',kind='transition')
        row=dict(identity=identity,projection=target,annotation_trace=audit,
                 original_stride16_previous_intent=anchors[8][2],human_admission=False,
                 teacher_kind='annotation_periodic_command_trace_not_actor_rollout')
        return row,episode

    def test_candidate_rebuilt_from_same_frame_not_dense_previous(self):
        row,episode=self.candidate()
        result=module.validate_candidate_projection(row,episode,original_index=100,original_pair=(7,8))
        self.assertEqual(result,row['projection'])
        self.assertNotEqual(result['previous_intent'],row['original_stride16_previous_intent'])
        self.assertFalse(result['outcome_supervision_mask'])
        self.assertFalse(result['low_action_supervision_mask'])

    def test_tampered_history_mask_identity_or_target_rejected(self):
        row,episode=self.candidate()
        for field,value in (('memory','future target'),('low_action_supervision_mask',True),
                            ('outcome_supervision_mask',True),('active_skills_text',self.command(target='other'))):
            changed=copy.deepcopy(row);changed['projection'][field]=value
            with self.assertRaises(ValueError):
                module.validate_candidate_projection(changed,episode,original_index=100,original_pair=(7,8))
        changed=copy.deepcopy(row);changed['identity']['frame']+=16
        with self.assertRaises(ValueError):
            module.validate_candidate_projection(changed,episode,original_index=100,original_pair=(7,8))
        with self.assertRaises(ValueError):
            module.validate_candidate_projection(row,dict(episode,split='eval'),original_index=100,original_pair=(7,8))

    def command(self, verb='GRASP', target='vacuum', source='', destination='', arm='UNSPECIFIED'):
        from g05.utils.memlite_skill_protocol import semantic_active_skills_text
        return semantic_active_skills_text([dict(verb=verb, target=target, source=source,
            destination=destination, target_part='', arm=arm, unbound_relation='')])

    def test_source_only_change_is_not_counted_as_skill_progress(self):
        self.assertEqual(module.command_change_class(self.command(source='floors'), self.command()),
                         'source_or_binding_description_change')
        self.assertEqual(module.command_change_class(self.command(target=''), self.command()),
                         'source_or_binding_description_change')

    def test_real_verb_target_or_destination_changes_remain_distinct(self):
        self.assertEqual(module.command_change_class(self.command(verb='NAVIGATE'), self.command()),
                         'explicit_verb_or_bundle_change')
        self.assertEqual(module.command_change_class(self.command(), self.command(target='broom')),
                         'explicit_bound_object_or_goal_change')
        self.assertEqual(module.command_change_class(self.command(verb='PLACE_ON', destination='table'),
                                                    self.command(verb='PLACE_ON', destination='shelf')),
                         'explicit_bound_object_or_goal_change')
        self.assertEqual(module.command_change_class('None', self.command()), 'initial')
        self.assertEqual(module.command_change_class(self.command(), self.command()), 'identical_command')

    def test_changed_unbound_entities_never_guess_target_roles(self):
        from g05.utils.memlite_skill_protocol import semantic_active_skills_text, canonical_json
        def text(entity):
            return semantic_active_skills_text([dict(verb='GRASP', target='', source='', destination='',
                target_part='', arm='UNSPECIFIED', unbound_relation=canonical_json(dict(entities=[entity, 'fridge'], spatial_relations=['layer4'])))])
        self.assertEqual(module.command_change_class(text('tupperware74'), text('tupperware75')),
                         'unbound_relation_change_not_verified_roles')

    def trace(self):
        segments = [dict(start=0, end=64, text='NAVIGATE', parent='P0'),
                    dict(start=64, end=192, text='GRASP', parent='P1'),
                    dict(start=192, end=512, text='PLACE', parent='P2')]
        rows = []
        previous = parent = 'None'
        for frame in range(0, 512, 16):
            index = next(i for i, s in enumerate(segments) if s['start'] <= frame < s['end'])
            rows.append((frame, index, previous, parent, []))
            previous, parent = segments[index]['text'], segments[index]['parent']
        return rows, segments

    def test_previous_intent_and_memory_recomputed_not_only_refresh_counter(self):
        anchors, segments = self.trace()
        records, audit = module.periodic_annotation_trace(anchors, segments)
        self.assertEqual([r['frame'] for r in records], [0, 128, 256, 384])
        self.assertEqual(anchors[8][2], 'GRASP')
        self.assertEqual(records[1]['previous_intent'], 'NAVIGATE')
        self.assertEqual(records[2]['previous_intent'], 'GRASP')
        self.assertEqual(records[2]['history'], ['NAVIGATE'])
        self.assertEqual(module.command_kind(records[1], segments[records[1]['segment']]), 'transition')
        self.assertEqual(module.command_kind(records[-1], segments[records[-1]['segment']]), 'continuation')
        self.assertIsNone(audit['first_unannotated_scheduled_call'])

    def test_annotation_gap_never_invents_previous_command(self):
        anchors, segments = self.trace()
        anchors = [row for row in anchors if row[0] != 128]
        records, audit = module.periodic_annotation_trace(anchors, segments)
        self.assertEqual(len(records), 1)
        self.assertEqual(audit['first_unannotated_scheduled_call'], 128)
        self.assertEqual(audit['omitted_known_calls_after_gap'], 2)

    def test_noncausal_clock_and_arbitrary_period_rejected(self):
        anchors, segments = self.trace()
        with self.assertRaises(ValueError): module.periodic_annotation_trace(anchors[::-1], segments)
        with self.assertRaises(ValueError): module.periodic_annotation_trace(anchors, segments, period=16)


if __name__ == '__main__': unittest.main()

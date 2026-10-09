from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_history import replay_episode, join_causal_history
from recovery_corpus import canonical, group_key
from g05.utils.memlite_skill_protocol import canonical_json, append_b_memory_idempotent, semantic_active_skills_text


def fixture():
    ep = dict(run='run', episode_id='episode', task='sweeping_garage', instance_id=211,
              source_commit='code', high_checkpoint_sha256='high', low_checkpoint_sha256='low', gpu=0)
    skills = canonical([dict(verb='NAVIGATE', target='broom', source='', destination='',
                            target_part='', unbound_relation='', arm='UNSPECIFIED')])
    text = semantic_active_skills_text(json.loads(skills))
    memory = canonical_json(dict(task_name='sweeping garage', issued_command_history=[], verified_world_facts=[]))
    previous = 'None'
    logs = []
    for i in range(3):
        expected = append_b_memory_idempotent(memory, previous, task_name='sweeping garage')
        projection = dict(schema_version=6, memlite_branch='low', task_name='sweeping garage',
            parent_goal='Task goal: sweeping garage', target_parent_goal='', previous_parent_goal='none',
            memory=memory, previous_intent='None', known_previous_outcome='UNKNOWN', execution_feedback='none',
            active_skills_semantic_json=skills, active_skills_text=text, next_decision='EXECUTE', memory_update='',
            task_complete=False, outcome_target='UNKNOWN', outcome_supervision_mask=False,
            parent_goal_supervision_mask=False, low_action_supervision_mask=False)
        logs.append(dict(episode=ep, task='sweeping garage', rank=0, control_step=i*128, chunk=i*8,
            context_id=hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest(),
            event=dict(validated=True, previous_outcome='UNKNOWN', task_complete=False,
                task_complete_claimed=False, decision='EXECUTE', parent_goal=projection['parent_goal'],
                active_skills_semantic_json=skills, active_skills_text=text, memory_update=expected)))
        memory, previous = expected, text
    return ep, logs


class HistoryTests(unittest.TestCase):
    def test_exact_recurrence_and_repeat_age(self):
        ep, logs = fixture()
        out = replay_episode(logs, ep)
        self.assertEqual([x['intent_started_control_step'] for x in out], [0, 0, 0])
        self.assertEqual([x['repeated_planning_count'] for x in out], [0, 1, 2])
        logs[2]['event']['decision'] = 'RETRY'
        out = replay_episode(logs, ep)
        self.assertEqual(out[2]['intent_started_control_step'], 256)
        self.assertEqual(out[2]['attempt_number'], 2)

    def test_missing_cross_episode_mutated_memory_and_hash_rejected(self):
        ep, logs = fixture()
        bads = [logs[1:]]
        for key, value in [('context_id', 'wrong'), ('rank', 1)]:
            bad = deepcopy(logs); bad[1][key] = value; bads.append(bad)
        bad = deepcopy(logs); bad[1]['event']['memory_update'] = '{}'; bads.append(bad)
        bad = deepcopy(logs); bad[1]['episode']['instance_id'] = 99; bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError): replay_episode(bad, ep)

    def test_h1_does_not_read_own_target_or_future(self):
        ep, logs = fixture()
        anchors, raw = [], {}
        for t in [0, 16, 128, 144]:
            log = logs[t//128]
            anchors.append(dict(sample_id=str(t), source_episode=['run', 'episode'],
                source_group=group_key(ep['task'], 211), control_step=t,
                actor_input=dict(issued_skills_semantic_json=log['event']['active_skills_semantic_json'],
                                 parent_goal=log['event']['parent_goal'])))
            raw[t] = dict(context=dict(context_id=log['context_id']))
        class Reader:
            def episode(self, _): return raw, {}
        out = join_causal_history(anchors, [dict(episode=ep)], logs, Reader())
        self.assertIsNone(out[0]['predecision'])
        self.assertIsNone(out[1]['predecision'])
        self.assertEqual(out[2]['observable']['control_step'], 128)
        self.assertEqual(out[2]['predecision']['control_step'], 0)
        self.assertEqual(out[2]['predecision']['served_controls'], 128)
        self.assertEqual(out[3]['observable']['control_step'], 128)
        self.assertNotIn('physical_audit', json.dumps(out))


if __name__ == '__main__': unittest.main()

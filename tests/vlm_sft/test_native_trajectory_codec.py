from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
import native_trajectory_codec as codec


class CompositeCodecTests(unittest.TestCase):
    def example(self):
        state = np.zeros(61)
        state[23] = state[48] = 1
        state[53:57] = [.1, -.2, .3, -.4]
        state[3:10] = np.arange(7) * .1
        state[28:35] = -np.arange(7) * .15
        q = codec.current_anchor(state)
        a = np.zeros((16, 23))
        for k, sl in codec.JOINT_ACTION.items():
            a[:, sl] = q[codec.JOINT_ANCHOR[k]]
        a[:, [14, 22]] = 1
        a[:, 0] = np.linspace(0, .4, 16)
        a[:, 2] = -.1
        a[:, 4] += np.linspace(0, .018, 16)
        a[:, 7] += np.linspace(.005, -.035, 16)
        a[:, 18] += np.sin(np.linspace(0, 1, 16)) * .075
        a[5:, 14] = -1
        a[12:, 22] = -1
        return state, a

    def test_all_controls_roundtrip_and_gripper_event_clock(self):
        state, a = self.example()
        text = codec.encode(a, state)
        decoded = codec.decode(text, codec.current_anchor(state))
        self.assertEqual(decoded.shape, (16, 23))
        for k, v in codec.errors(a, decoded).items():
            self.assertLessEqual(v, codec.ERROR_LIMITS[k] + 1e-10)
        np.testing.assert_array_equal(decoded[:, [14, 22]], a[:, [14, 22]])
        self.assertEqual([r[0] for r in json.loads(text)['g']], [0, 5, 12])
        # This would fail if left/right or torso/native27 padding were mixed up.
        np.testing.assert_allclose(decoded[:, 3], a[:, 3], atol=1e-10)
        np.testing.assert_allclose(decoded[:, 17], a[:, 17], atol=1e-10)

    def test_joint_offsets_use_current_anchor_not_previous_knot(self):
        state, a = self.example()
        text = codec.encode(a, state)
        q = codec.current_anchor(state)
        shifted = q.copy(); shifted[4] += .2
        original = codec.decode(text, q)
        other = codec.decode(text, shifted.round(6))
        np.testing.assert_allclose(other[:, 7] - original[:, 7], .2, atol=1e-12)
        np.testing.assert_allclose(other[:, 8:], original[:, 8:])
        with self.assertRaises(ValueError):
            codec.decode(text, q + .0000001)

    def test_alternating_and_curved_targets_cannot_hide_middle_error(self):
        state, a = self.example()
        a[:, 7] += np.where(np.arange(16) % 2, .1, -.1)
        obj = json.loads(codec.encode(a, state))
        self.assertEqual(len(obj['l']), 16)
        decoded = codec.decode(json.dumps(obj), codec.current_anchor(state))
        self.assertLessEqual(codec.errors(a, decoded)['joint_rad'], codec.ERROR_LIMITS['joint_rad'])

    def test_constant_groups_are_single_knots(self):
        state, a = self.example()
        a[:] = a[0]
        obj = json.loads(codec.encode(a, state))
        self.assertTrue(all(len(obj[k]) == 1 for k in codec.GROUP_WIDTHS))

    def test_every_gripper_change_preserved_including_tick_fifteen(self):
        state, a = self.example()
        a[:, 14] = np.where(np.arange(16) % 2, -.5, 1.)
        obj = json.loads(codec.encode(a, state))
        self.assertEqual(len(obj['g']), 16)
        np.testing.assert_array_equal(codec.decode(json.dumps(obj), codec.current_anchor(state))[:, 14], a[:, 14])

    def test_native_control_packing_equivalence(self):
        from semantic_robot.v2.servo import native_action
        state, a = self.example()
        b = codec.decode(codec.encode(a, state), codec.current_anchor(state))
        for row in b:
            q = np.r_[row[3:7], row[7:14], row[15:22]]
            packed = native_action(q, row[[14, 22]], row[:3] * [.75, .75, 1.])
            np.testing.assert_array_equal(packed, row.astype(np.float32))

    def test_bad_json_and_untrusted_fields_rejected(self):
        state, a = self.example(); obj = json.loads(codec.encode(a, state))
        bad = [True, [], {**obj, 'v': 'wrong'}, {**obj, 'future': []}]
        for key in codec.GROUP_WIDTHS:
            missing = deepcopy(obj); del missing[key]; bad.append(missing)
        for field, value in [('t', [[0, 0, 0, 0, True]]), ('l', [[0, 1.1, 0, 0, 0, 0, 0, 0]]),
                             ('b', [[0, 1001, 0, 0]]), ('g', [[0, 1, 1001]]),
                             ('r', [[1, 0, 0, 0, 0, 0, 0, 0]]), ('g', [[0, 1, 1], [0, -1, -1]]),
                             ('t', [[0, 0, 0, 0, 0], [14, 0, 0, 0, 0]]),
                             ('b', [[0, 0, 0, 0], [16, 0, 0, 0]])]:
            bad.append({**obj, field: value})
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): codec.parse(json.dumps(value))
        for text in ['{"v":1,"v":2}', '{"v":NaN}', '```json\n{}\n```', 'x' * 32769]:
            with self.assertRaises(ValueError): codec.parse(text)

    def test_source_shape_nonfinite_and_range_rejected(self):
        state, a = self.example()
        for bad in [a[:, :-1], np.full((16, 23), np.nan), np.ones((16, 23), dtype=bool)]:
            with self.assertRaises(ValueError): codec.encode(bad, state)
        a[3, 1] = 1.1
        with self.assertRaises(ValueError): codec.encode(a, state)

    def test_actor_is_causal_and_anchor_identical(self):
        state, a = self.example()
        actor = codec.actor_from_state('Put the plate in the fridge.', 'verb=PLACEIN; target=plate', state)
        text = codec.prompt(actor)
        self.assertNotIn('episode', text)
        np.testing.assert_array_equal(actor['proprio']['q_rad'], codec.current_anchor(state))
        original = codec.prompt(actor)
        a[:] = 999
        self.assertEqual(original, codec.prompt(actor))
        for key in ('future_state', 'source_id', 'success', 'expert_action'):
            with self.assertRaises(ValueError): codec.prompt({**actor, key: True})
        actor['proprio']['future_q'] = [1.] * 18
        with self.assertRaises(ValueError): codec.prompt(actor)


if __name__ == '__main__': unittest.main()

"""CPU protocol checks; fixtures are synthetic, never success-rate evidence."""

from copy import deepcopy
import unittest

import numpy as np

from experiments.zetta_behavior.bridge import (
    ACTION_PARTS,
    FrozenG05ClientBoundary,
    action23_from_g05,
    inspect_rule_requirements,
)


def response():
    cursor = 0
    action = {}
    for key, width in ACTION_PARTS:
        action[key] = np.arange(cursor, cursor + width, dtype=np.float32) / 100
        cursor += width
    return {"action": action, "need_obs": False, "cot_text": "unused"}


def observation():
    return {"images": {"head_rgb": "synthetic"}, "state": {}, "task": "test"}


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.reply = response()
        self.reset_reply = {"__reset__": True}

    def __call__(self, message):
        self.calls.append(deepcopy(message))
        if message == {"__reset__": True}:
            return self.reset_reply
        return self.reply


class ActionContractTests(unittest.TestCase):
    def test_keeps_all_23_dims_including_base_and_torso(self):
        actual = action23_from_g05(response())
        np.testing.assert_array_equal(actual, np.arange(23, dtype=np.float32) / 100)
        self.assertEqual(actual.dtype, np.float32)

    def test_does_not_mutate_or_alias_the_reply(self):
        reply = response()
        before = deepcopy(reply)
        value = action23_from_g05(reply)
        value[:] = 99
        for key in reply["action"]:
            np.testing.assert_array_equal(reply["action"][key], before["action"][key])

    def test_every_missing_part_is_an_error(self):
        for key, _ in ACTION_PARTS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                reply = response()
                del reply["action"][key]
                action23_from_g05(reply)

    def test_rejects_errors_27d_and_unmerged_protocols(self):
        for reply in (None, {"error": {"code": 500}}, {"action": np.zeros(27)},
                      {"action": {"left_control": np.zeros(9)}}):
            with self.subTest(reply=str(reply)), self.assertRaises(ValueError):
                action23_from_g05(reply)

    def test_rejects_bad_shapes_types_nonfinite_and_overflow(self):
        bad = (np.zeros((1, 3)), np.zeros((32, 3)), [0, 1],
               [True, False, True], ["0", "1", "2"],
               [0, np.nan, 0], [0, np.inf, 0], [0, 1e99, 0],
               [0j, 1j, 2j])
        for value in bad:
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                reply = response()
                reply["action"]["base_qvel"] = value
                action23_from_g05(reply)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.client = FrozenG05ClientBoundary(self.transport)

    def test_requires_initial_reset_before_requesting_actions(self):
        with self.assertRaises(RuntimeError):
            self.client.policy_step(observation(), step_index=0)
        self.assertEqual(self.transport.calls, [])

    def test_handoff_discards_remote_chunk_and_resumes_at_new_tick(self):
        self.client.reset_episode()
        self.client.policy_step(observation(), step_index=0)
        self.client.recovery_started()
        with self.assertRaises(RuntimeError):
            self.client.policy_step(observation(), step_index=1)
        self.client.recovery_finished()
        self.client.policy_step(observation(), step_index=21)
        self.assertEqual(len(self.transport.calls), 4)
        self.assertEqual(self.transport.calls[2], {"__reset__": True})
        self.assertEqual(self.transport.calls[3], observation())

    def test_recovery_does_not_rewind_episode_action_clock(self):
        self.client.reset_episode()
        self.client.policy_step(observation(), step_index=5)
        self.client.recovery_started()
        self.client.recovery_finished()
        with self.assertRaises(ValueError):
            self.client.policy_step(observation(), step_index=5)

    def test_rejects_invalid_ticks_without_io(self):
        self.client.reset_episode()
        for step in (-1, 1.5, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.client.policy_step(observation(), step_index=step)
        self.assertEqual(len(self.transport.calls), 1)

    def test_failed_reset_stays_blocked(self):
        self.transport.reset_reply = {"error": "bad"}
        with self.assertRaises(ValueError):
            self.client.reset_episode()
        with self.assertRaises(RuntimeError):
            self.client.policy_step(observation(), step_index=0)

    def test_bad_reply_stays_blocked_and_cannot_reuse_cached_suffix(self):
        self.client.reset_episode()
        self.transport.reply = {"error": "bad"}
        with self.assertRaises(ValueError):
            self.client.policy_step(observation(), step_index=0)
        self.transport.reply = response()
        with self.assertRaises(RuntimeError):
            self.client.policy_step(observation(), step_index=1)

    def test_transport_exception_stays_blocked(self):
        self.client.reset_episode()
        def unavailable(_):
            raise TimeoutError("synthetic timeout")
        self.client._request = unavailable
        with self.assertRaises(TimeoutError):
            self.client.policy_step(observation(), step_index=0)
        with self.assertRaises(RuntimeError):
            self.client.policy_step(observation(), step_index=1)

    def test_resetting_episode_clears_clock(self):
        self.client.reset_episode()
        self.client.policy_step(observation(), step_index=200)
        self.client.reset_episode()
        self.client.policy_step(observation(), step_index=0)

    def test_forbids_empty_cached_or_reset_as_policy_request(self):
        self.client.reset_episode()
        for obs in ({}, {"task": "test"}, {**observation(), "__reset__": True}):
            with self.subTest(obs=obs), self.assertRaises(ValueError):
                self.client.policy_step(obs, step_index=0)
        self.assertEqual(len(self.transport.calls), 1)


class FeatureAuditTests(unittest.TestCase):
    def test_privileged_guard_cannot_hide_behind_public_primary_feature(self):
        payload = {"critic_rules": [{
            "feature": "command.available",
            "activation_conditions": [{"feature": "privileged.task.success"}],
        }]}
        result = inspect_rule_requirements(payload, public_features=frozenset({
            "command.available", "privileged.task.success",
        }))
        self.assertFalse(result["public_observation_compatible"])
        self.assertEqual(result["privileged_features"], ["privileged.task.success"])

    def test_missing_features_are_not_filled_with_zero(self):
        payload = {"critic_rules": [{"feature": "vision.object.progress"}]}
        result = inspect_rule_requirements(payload, public_features=frozenset())
        self.assertEqual(result["unavailable_public_features"], ["vision.object.progress"])
        self.assertFalse(result["public_observation_compatible"])

    def test_valid_feature_report_is_not_a_recovery_admission(self):
        payload = {"critic_rules": [{"feature": "command.available"}]}
        result = inspect_rule_requirements(payload, public_features=frozenset({"command.available"}))
        self.assertTrue(result["public_observation_compatible"])
        self.assertNotIn("ready_for_recovery", result)

    def test_malformed_rules_rejected(self):
        for payload in ({}, {"critic_rules": {}}, {"critic_rules": [None]},
                        {"critic_rules": [{"feature": ""}]},
                        {"critic_rules": [{"feature": "x", "activation_conditions": [None]}]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                inspect_rule_requirements(payload, public_features=frozenset())


if __name__ == "__main__":
    unittest.main()

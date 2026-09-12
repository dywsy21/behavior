from types import SimpleNamespace

import pytest
import numpy as np

from scripts.data.package_memlite_recovery import initial_intent, validate_approach_dynamics, RecoveryQualityRejected


def test_initial_intent_can_have_unannotated_lead_in_without_future_completion():
    rows = {198: [{"primitive_idx": 0, "intent": "open door [cabinet]", "ignore": False}],
            700: [{"primitive_idx": 1, "intent": "close door [cabinet]"}]}
    side = SimpleNamespace(_load_episode=lambda episode: rows)
    assert initial_intent(side, 800, "cabinet") == ("open door [cabinet]", 198)


@pytest.mark.parametrize("row", [
    {"primitive_idx": None, "intent": "open [cabinet]"},
    {"primitive_idx": 1, "intent": "close [cabinet]"},
    {"primitive_idx": 0, "intent": "open [other]"},
    {"primitive_idx": 0, "intent": "open [cabinet]", "ignore": True},
    {"primitive_idx": 0, "intent": "open [cabinet]", "memory_input_corruption": "false_completion"},
])
def test_initial_intent_rejects_wrong_stage_target_and_corruption(row):
    side = SimpleNamespace(_load_episode=lambda episode: {0: [row]})
    with pytest.raises(ValueError, match="Missing initial primitive"):
        initial_intent(side, 800, "cabinet")


def test_successful_endpoint_does_not_hide_uncommanded_approach_rotation():
    state, action = np.zeros((32,61)), np.zeros((32,23))
    state[5,2] = .67
    with pytest.raises(RecoveryQualityRejected, match="Uncommanded"):
        validate_approach_dynamics(state, action, ["approach"]*16+["settle"]*16)
    validate_approach_dynamics(state, action, ["perturb"]*16+["settle"]*16)
    state[5,2] = .05
    validate_approach_dynamics(state, action, ["approach"]*16+["settle"]*16)

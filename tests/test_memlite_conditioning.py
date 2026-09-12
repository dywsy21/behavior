from copy import deepcopy

import pytest
import torch

from g05.models.g05.helpers.memlite_conditioning import MEMLiteConditioning


def sample(branch="low"):
    return {"memlite_branch": branch, "template": "Task: <command_text_!> State: <proprio_proprio_!>; Action: <EOC><action_action>",
            "proprio": {"value": torch.arange(162, dtype=torch.float32).reshape(6, 27),
                        "proprio_dim_is_pad": torch.zeros(27, dtype=torch.bool)},
            "action": {"value": torch.ones(32, 27)}, "intent": "Face the radio"}


def test_disabled_preserves_legacy_inputs():
    rows = [sample()]
    assert MEMLiteConditioning().prepare(rows, training=True) is rows


def test_mask_is_input_only_all_history_and_explicit_unknown():
    original = sample()
    before = deepcopy(original)
    result = MEMLiteConditioning(enabled=True, velocity_dropout=1).prepare([original], training=True)[0]
    torch.testing.assert_close(original["proprio"]["value"], before["proprio"]["value"])
    assert torch.equal(result["proprio"]["value"][:, :24], before["proprio"]["value"][:, :24])
    assert not result["proprio"]["value"][:, 24:].any()
    assert result["action"] is original["action"]
    assert "unknown, not zero speed" in result["base_velocity_observation"]
    assert "<execution_feedback_text_!>" in result["template"]
    assert "<base_velocity_observation_text_!>" in result["template"]
    assert result["execution_feedback"] == "none"


@pytest.mark.parametrize("training", [True, False])
def test_high_level_retains_real_velocity_for_progress_observation(training):
    row = sample("high")
    result = MEMLiteConditioning(enabled=True, velocity_dropout=1).prepare([row], training=training)[0]
    assert result["proprio"] is row["proprio"]
    assert result["base_velocity_observation"] == "observed"


def test_train_and_infer_masked_contract_identical():
    c = MEMLiteConditioning(enabled=True, velocity_dropout=1, inference_velocity="masked")
    a = c.prepare([sample()], training=True)[0]
    b = c.prepare([sample()], training=False)[0]
    assert a["template"] == b["template"] and a["base_velocity_observation"] == b["base_velocity_observation"]
    torch.testing.assert_close(a["proprio"]["value"], b["proprio"]["value"])
    again = c.prepare([b], training=False)[0]
    assert again["template"].count("Base velocity observation:") == 1


def test_observed_and_override_modes_do_not_overwrite_source():
    c = MEMLiteConditioning(enabled=True, velocity_dropout=0)
    s = sample()
    assert c.prepare([s], training=True)[0]["proprio"] is s["proprio"]
    s["memlite_velocity_override"] = "observed"
    assert c.prepare([s], training=False)[0]["proprio"] is s["proprio"]


@pytest.mark.parametrize("config", [{"velocity_dropout": -1}, {"velocity_dropout": float("nan")},
    {"inference_velocity": "zero"}, {"velocity_indices": [1, 1, 2]}, {"velocity_indices": [25, 26, 27]}])
def test_bad_config_is_rejected(config):
    with pytest.raises(ValueError):
        MEMLiteConditioning.from_config(config)


def test_no_silent_wrong_embodiment_or_missing_state():
    s = sample()
    s["proprio"]["value"] = torch.zeros(6, 23)
    with pytest.raises(ValueError, match="Expected"):
        MEMLiteConditioning(enabled=True).prepare([s], training=False)

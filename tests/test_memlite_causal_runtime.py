import asyncio
from copy import deepcopy
import numpy as np
import pytest
import torch

from scripts.serve_policy_mem import ChunkedPolicyWrapper


DIMS = {"base_qvel": 3, "trunk_qpos": 4, "left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1}


class Processor:
    num_obs_steps = 6
    memlite_runtime = dict(schema_version=5, action_steps=2, replan_every_chunks=8,
                           high_level_max_new_tokens=1024, terminal_recheck_every_chunks=2)
    shape_meta = {"images": [{"key": "cam", "raw_shape": [3, 2, 2]}],
                  "state": [{"key": key, "raw_shape": 2 if key.endswith("gripper") else dim} for key, dim in DIMS.items()],
                  "action": [{"key": key, "raw_shape": dim} for key, dim in DIMS.items()]}


def obs():
    return {"images": {"cam": np.zeros((3, 2, 2), dtype=np.uint8)},
            "state": {key: np.full(2 if key.endswith("gripper") else dim, .025, dtype=np.float32) for key, dim in DIMS.items()},
            "task": "prepare cup"}


def proposal(intent="prepare cup", status="CONTINUE"):
    return {"raw": f"Intent: {intent}|Updated Memory: Task=0; Completed=none.|Status: {status}|<HL_END>"}


class Inferencer:
    policy = type("Policy", (), {"predict_cot": True})()

    def __init__(self, proposals=None):
        self.proposals = list(proposals or [proposal()])
        self.high_obs, self.low_intents = [], []
        self.actions = {key: torch.zeros(1, 2, dim) for key, dim in DIMS.items()}
        self.actions["left_gripper"].fill_(-1)
        self.actions["right_gripper"].fill_(1)

    def infer_high_level(self, observations, memories, *, max_new_tokens):
        self.high_obs.append(deepcopy(observations[0]))
        assert max_new_tokens == 1024
        item = self.proposals.pop(0) if self.proposals else proposal()
        if isinstance(item, Exception):
            raise item
        return [item]

    def infer_low_level_action(self, observations, intents):
        self.low_intents.extend(intents)
        return [deepcopy(self.actions)]


def wrapper(inferencer):
    w = ChunkedPolicyWrapper(inferencer, Processor(), action_steps=2)
    w.initialize_memlite_memory(task_id=0, canonical_memory="Task=0; Completed=none.")
    return w


def step(w, observation=None):
    return asyncio.run(w.get_action(obs() if observation is None else observation))[0]


def test_done_holds_without_decoding_actions_and_rechecks():
    inf = Inferencer([proposal("Task complete", "DONE"), proposal()])
    w = wrapper(inf)
    step(w)
    assert not inf.low_intents
    assert w.mem_state.status == "DONE"
    for _ in range(3):
        step(w)
    assert len(inf.high_obs) == 1 and not inf.low_intents
    step(w)
    assert len(inf.high_obs) == 2 and inf.low_intents == ["prepare cup"]
    assert inf.high_obs[-1]["previous_intent"] == "Task complete"


def test_malformed_proposal_is_not_committed_and_preserves_gripper_command():
    inf = Inferencer([proposal(), {"raw": "Intent: bad|Status: CONTINUE"}])
    w = wrapper(inf)
    step(w)
    old_memory = w.mem_state.memory_text
    old_intent = w.mem_state.intent_text
    hold = step(w, {**obs(), "replan": True})
    assert w.mem_state.memory_text == old_memory and w.mem_state.intent_text == old_intent
    assert len(inf.low_intents) == 1 and w._high_level_pending
    assert float(hold["left_gripper"][0]) == -1 and float(hold["right_gripper"][0]) == 1
    assert not np.any(hold["base_qvel"])
    w.reset()
    assert not w._last_gripper_commands and w._terminal_hold_chunks == 0


@pytest.mark.parametrize("failure", ["missing_gripper", "extreme_inverse", "extreme_target"])
def test_invalid_actions_hold_and_request_fresh_plan(failure):
    inf = Inferencer()
    if failure == "missing_gripper":
        inf.actions.pop("left_gripper")
    elif failure == "extreme_inverse":
        inf.actions["_normalization_diagnostics"] = {"left_arm": {"max_normalized_excess": 100.0}}
    else:
        inf.actions["left_arm"].fill_(1000)
    w = wrapper(inf)
    action = step(w)
    assert w._high_level_pending and w.mem_state.failure_count == 1
    np.testing.assert_allclose(action["left_arm"], obs()["state"]["left_arm"])


def test_infrastructure_errors_are_not_hidden_by_hold():
    w = wrapper(Inferencer([RuntimeError("CUDA out of memory")]))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        step(w)


def test_causal_runtime_requires_initialized_task_memory():
    w = ChunkedPolicyWrapper(Inferencer(), Processor(), action_steps=2)
    with pytest.raises(ValueError, match="Initialize canonical task memory"):
        step(w)


def test_runtime_rejects_wrong_action_history_cadence():
    with pytest.raises(ValueError, match="history cadence"):
        ChunkedPolicyWrapper(Inferencer(), Processor(), action_steps=16)

"""Runtime anchors must not be reconstructed from lossy normalized proprio."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from g05.data_processor.transforms.relative_action import BehaviorPerKeyTransform
from g05.data_processor.processor.base_processor import ActionProcessor
from g05.models.g05.inferencer import PolicyInferencer


def packet(batch_size=1, horizon=4, anchored=True):
    state = {"left_arm": torch.zeros(batch_size, 2, 7),
             "right_arm": torch.full((batch_size, 2, 7), -1.41068184),
             "lower_body": torch.zeros(batch_size, 2, 7)}
    action = {"left_arm": torch.full((batch_size, horizon, 7), .1),
              "right_arm": torch.full((batch_size, horizon, 7), .2),
              "lower_body": torch.full((batch_size, horizon, 7), .3)}
    data = {"state": state, "action": action}
    if anchored:
        data["_raw_state_anchor"] = {
            "left_arm": torch.arange(batch_size, dtype=torch.float32)[:, None, None].expand(-1, 1, 7).clone(),
            "right_arm": torch.full((batch_size, 1, 7), -2.75474286),
            "trunk_qpos": torch.full((batch_size, 1, 4), .4)}
    return data


def test_raw_anchor_preserves_real_pose_and_batch_identity():
    data = packet(batch_size=2)
    before = deepcopy(data["_raw_state_anchor"])
    result = BehaviorPerKeyTransform().backward(data)
    torch.testing.assert_close(result["action"]["right_arm"], torch.full((2, 4, 7), -2.55474286))
    torch.testing.assert_close(result["action"]["left_arm"][0], torch.full((4, 7), .1))
    torch.testing.assert_close(result["action"]["left_arm"][1], torch.full((4, 7), 1.1))
    torch.testing.assert_close(result["action"]["trunk_qpos"][..., :3], torch.full((2, 4, 3), .7))
    # Fourth trunk coordinate and base velocity are absolute, not relative.
    torch.testing.assert_close(result["action"]["trunk_qpos"][..., 3], torch.full((2, 4), .3))
    torch.testing.assert_close(result["action"]["base_qvel"], torch.full((2, 4, 3), .3))
    for key in before:
        torch.testing.assert_close(data["_raw_state_anchor"][key], before[key])
    assert result["_state_anchor_diagnostics"]["right_arm"]["normalized_roundtrip_error_max"] == pytest.approx(1.34406102)


def test_legacy_no_anchor_is_unchanged():
    result = BehaviorPerKeyTransform().backward(packet(anchored=False))
    torch.testing.assert_close(result["action"]["right_arm"], torch.full((1, 4, 7), -1.21068184))
    assert "_state_anchor_diagnostics" not in result


@pytest.mark.parametrize("bad", ["missing", "nan", "wrong_dim", "missing_batch", "wrong_batch", "extra_time"])
def test_bad_anchor_is_rejected_not_silently_broadcast(bad):
    data = packet()
    if bad == "missing":
        del data["_raw_state_anchor"]["right_arm"]
    elif bad == "nan":
        data["_raw_state_anchor"]["right_arm"][0, 0, 1] = torch.nan
    else:
        shape = {"wrong_dim": (1, 1, 6), "missing_batch": (1, 7),
                 "wrong_batch": (2, 1, 7), "extra_time": (1, 2, 7)}[bad]
        data["_raw_state_anchor"]["right_arm"] = torch.zeros(shape)
    with pytest.raises(ValueError, match="anchor"):
        BehaviorPerKeyTransform().backward(data)


class MutatingProcessor:
    action_state_transforms = [BehaviorPerKeyTransform()]

    def preprocess(self, obs):
        for value in obs["state"].values():
            value.add_(99)
        return {"proprio": torch.zeros(2, 27)}


def test_prepare_captures_last_raw_state_before_mutation():
    obs = {"state": {key: torch.arange(3 * dim, dtype=torch.float32).reshape(3, dim)
                     for key, dim in (("left_arm", 7), ("right_arm", 7), ("trunk_qpos", 4))}}
    original = deepcopy(obs["state"])
    prepared = PolicyInferencer(None, MutatingProcessor(), "cpu")._prepare(obs)
    for key, value in original.items():
        torch.testing.assert_close(prepared.raw_state_anchor[key], value[-1:].unsqueeze(0))
        assert prepared.raw_state_anchor[key].shape[0:2] == (1, 1)
    assert "_raw_state_anchor" not in prepared.sample


def test_prepare_other_embodiment_does_not_assume_behavior_keys():
    processor = SimpleNamespace(preprocess=lambda obs: {"proprio": torch.ones(1, 2)})
    prepared = PolicyInferencer(None, processor, "cpu")._prepare({"state": {"unrelated": torch.ones(1, 2)}})
    assert prepared.raw_state_anchor is None


def test_postprocess_single_keeps_anchor_private_and_returns_diagnostics():
    anchor = packet()["_raw_state_anchor"]
    captured = {}

    def postprocess(data):
        captured.update(data)
        return {"action": {"left_arm": data["action"]},
                "_state_anchor_diagnostics": {"left_arm": {"source": "raw_observation"}}}

    processor = SimpleNamespace(postprocess=postprocess)
    result = PolicyInferencer._postprocess_single(
        {"action": torch.ones(2, 4, 7), "proprio": torch.zeros(2, 2, 7)}, 1, processor,
        raw_state_anchor=anchor)
    assert captured["_raw_state_anchor"] is anchor
    assert captured["action"].shape == (1, 4, 7)
    assert result["_state_anchor_diagnostics"]["left_arm"]["source"] == "raw_observation"
    assert "_raw_state_anchor" not in result


class Identity:
    @staticmethod
    def backward(data):
        return data


@pytest.mark.parametrize("branch", ["action", "action_ar", "action_fm"])
def test_all_postprocess_routes_carry_raw_anchor(branch):
    original = packet()
    facade = SimpleNamespace(action_state_merger=Identity(), normalizer=Identity(),
                             action_state_transforms=[BehaviorPerKeyTransform()], action_filter=Identity(),
                             _slice_execution_actions=lambda action: action)
    data = {branch: original["action"], "proprio": original["state"],
            "_raw_state_anchor": original["_raw_state_anchor"]}
    result = ActionProcessor.postprocess(facade, data)
    torch.testing.assert_close(result["action"]["right_arm"], torch.full((1, 4, 7), -2.55474286))
    assert result["_state_anchor_diagnostics"]["right_arm"]["source"] == "raw_observation"


def test_memlite_low_batch_keeps_anchors_out_of_model_and_matches_samples():
    anchors = [packet()["_raw_state_anchor"], packet()["_raw_state_anchor"]]
    anchors[1]["right_arm"].fill_(-.5)
    prepared = [SimpleNamespace(sub_processor=object(), raw_state_anchor=anchor) for anchor in anchors]
    model_inputs, routed = [], []

    class Policy:
        def generate_low_level_action(self, **kwargs):
            model_inputs.append(kwargs)
            return {"action": torch.zeros(2, 4, 27)}

    inferencer = PolicyInferencer(Policy(), None, "cpu")
    inferencer._prepare_branch_batch = lambda obs: (prepared, {
        "samples": [{"intent": "first"}, {"intent": "second"}], "pixel_values": {},
        "proprio": torch.zeros(2, 2, 27)})

    def postprocess(batch, index, processor, *, raw_state_anchor):
        routed.append((index, raw_state_anchor))
        return {"sample_index": index}

    inferencer._postprocess_single = postprocess
    assert inferencer.infer_low_level_action([{}, {}], ["first", "second"]) == [{"sample_index": 0}, {"sample_index": 1}]
    assert routed[0][1] is anchors[0] and routed[1][1] is anchors[1]
    assert all("_raw_state_anchor" not in sample for sample in model_inputs[0]["samples"])


def test_server_records_anchor_diagnostics_without_treating_them_as_actions():
    from test_memlite_causal_runtime import Inferencer, wrapper, step, DIMS

    inferencer = Inferencer()
    inferencer.actions["_state_anchor_diagnostics"] = {"right_arm": {
        "source": "raw_observation", "normalized_roundtrip_error_max": 1.34406102}}
    policy = wrapper(inferencer)
    events = []
    policy._trace = lambda event, **fields: events.append((event, fields))
    first = step(policy)
    second = step(policy)
    assert set(first) == set(DIMS) and set(second) == set(DIMS)
    assert set(policy._cached_chunk) == set(DIMS)
    anchor_events = [fields for event, fields in events if event == "state_anchor"]
    assert len(anchor_events) == 1
    assert anchor_events[0]["parts"]["right_arm"]["normalized_roundtrip_error_max"] == pytest.approx(1.34406102)


@pytest.mark.parametrize("diagnostics", ["none", "anchor", "normalization", "both"])
def test_generic_server_keeps_inferencer_diagnostics_out_of_action_cache(monkeypatch, diagnostics):
    import asyncio
    import numpy as np
    from scripts import serve_policy as serve

    # Exercise the real generic cache path without constructing a VLM or a
    # camera processor. The same inferencer is shared by both policy servers.
    monkeypatch.setattr(serve, "_parse_task_and_plan", lambda observation: None)
    monkeypatch.setattr(serve, "build_obs_dict", lambda observation, processor: observation)
    actions = {"right_arm": torch.arange(14, dtype=torch.float32).reshape(1, 2, 7)}
    if diagnostics in ("anchor", "both"):
        actions["_state_anchor_diagnostics"] = {"right_arm": {"source": "raw_observation"}}
    if diagnostics in ("normalization", "both"):
        actions["_normalization_diagnostics"] = {"right_arm": {"max_normalized_excess": 0.0}}
    calls = []

    def infer(observations):
        calls.append(observations)
        return [deepcopy(actions)]

    policy = serve.ChunkedPolicyWrapper(SimpleNamespace(infer=infer), None, action_steps=2)
    first, _ = asyncio.run(policy.get_action({"task": "prepare cup"}))
    second, _ = asyncio.run(policy.get_action({}))
    assert len(calls) == 1
    assert set(first) == set(second) == set(policy._cached_chunk) == {"right_arm"}
    np.testing.assert_array_equal(first["right_arm"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(second["right_arm"], np.arange(7, 14, dtype=np.float32))

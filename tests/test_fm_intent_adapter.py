from copy import deepcopy
import itertools
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from g05.models.g05.helpers.fm_intent_adapter import FMIntentAdapter
from g05.models.g05.g05_policy_memlite_fm import G05PolicyMEMLiteFM
from g05.models.g05.memlite_fm_inferencer import latest_observation, SeparatedMEMLiteFMInferencer
from g05.data_processor.processor.samples_builder import BaseSamplesBuilder, FMIntentActionBuilder
from g05.utils.common.task_event_sampler import build_memlite_train_sampler
from test_motion_sampler import View


def adapter():
    torch.manual_seed(71)
    return FMIntentAdapter(token_dim=12, feature_dim=8, action_dim=3,
        hidden_dim=16, num_heads=4, num_layers=1, max_tokens=12)


def test_preserved_model_stays_eval_with_non_module_tokenizer_wrapper():
    policy = G05PolicyMEMLiteFM.__new__(G05PolicyMEMLiteFM)
    nn.Module.__init__(policy)
    policy.model = nn.Sequential(nn.Linear(8,8), nn.Dropout(.5))
    policy.model.requires_grad_(False)
    policy.action_tokenizer = SimpleNamespace()
    policy.fm_intent_adapter = adapter()
    policy.train()
    assert policy.training and policy.fm_intent_adapter.training and not policy.model.training
    policy.eval()
    assert not policy.training and not policy.fm_intent_adapter.training and not policy.model.training
    groups = policy.get_optim_param_groups(.001, .01)
    assert {id(p) for g in groups for p in g["params"]} == {id(p) for p in policy.fm_intent_adapter.parameters()}


def inputs():
    return torch.randn(2, 5, 12), torch.tensor([[1,1,0,0,0],[1,1,1,1,1]]).bool(), torch.tensor([True, False])


def test_zero_initialization_then_real_gradients_without_touching_frozen_inputs():
    a = adapter()
    embeddings, mask, active = inputs()
    embeddings.requires_grad_(True)
    features = torch.randn(2, 4, 8, requires_grad=True)
    out = a(features, a.encode(embeddings, mask, active))
    assert torch.equal(out, torch.zeros_like(out))
    optimizer = torch.optim.AdamW(a.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        out = a(features, a.encode(embeddings, mask, active))
        (out - 1).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
        optimizer.step()
    assert embeddings.grad is None and features.grad is None
    assert out[0].abs().sum() > 0
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert a.token_projection.weight.grad.abs().sum() > 0


def test_padding_is_ignored_and_flow_sample_batch_order_is_correct():
    a = adapter().eval()
    nn.init.normal_(a.output.weight, std=.1)
    embeddings, mask, active = inputs()
    changed = embeddings.clone()
    changed[~mask] = 10000
    features = torch.randn(2, 4, 8)
    expected = a(features, a.encode(embeddings, mask, active))
    actual = a(features, a.encode(changed, mask, active))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tiled = a(features.repeat(3,1,1), a.encode(embeddings, mask, active))
    torch.testing.assert_close(tiled, expected.repeat(3,1,1))
    assert torch.equal(tiled[1::2], torch.zeros_like(tiled[1::2]))


def test_context_rejects_empty_mask_overlength_and_misaligned_flow_batch():
    a = adapter()
    with pytest.raises(ValueError, match="token"):
        a.encode(torch.zeros(1,2,12), torch.zeros(1,2).bool(), torch.ones(1).bool())
    with pytest.raises(ValueError, match="truncate"):
        a.encode(torch.zeros(1,13,12), torch.ones(1,13).bool(), torch.ones(1).bool())
    with pytest.raises(ValueError, match="multiple"):
        a(torch.zeros(3,4,8), a.encode(*inputs()))


def sample():
    return {"template": "SHOULD NOT REACH BASELINE", "image0": (256,256), "image1": (256,256),
            "image2": (256,256), "proprio": {"value": torch.arange(27).float().reshape(1,27)},
            "intent": "approach target", "memory": "old", "command": "turn on radio",
            "action": {"value": torch.ones(32,27)}, "memory_update": "FUTURE", "intent_status": "FUTURE",
            "execution_feedback": "spin", "memlite_branch": "low", "embodiment": "galaxea_r1pro"}


def test_baseline_prefix_has_no_intent_history_or_action_target_and_input_is_unmodified():
    source = sample()
    result = G05PolicyMEMLiteFM.baseline_samples([source])[0]
    expected = BaseSamplesBuilder(num_input_images=3, image_sizes={"a": (256,256)}).template
    assert result["template"].split("<EOC>")[0] == expected.split("<EOC>")[0]
    assert "action" not in result and "intent" not in result and "memory_update" not in result
    assert "<action_action" not in result["template"]
    assert source["memory_update"] == "FUTURE" and "action" in source
    torch.testing.assert_close(result["proprio"]["value"], source["proprio"]["value"], rtol=0, atol=0)
    source["proprio"]["value"] = torch.ones(6,27)
    with pytest.raises(ValueError, match="single latest"):
        G05PolicyMEMLiteFM.baseline_samples([source])


def test_fm_builder_keeps_intent_outside_baseline_and_rejects_high_or_terminal():
    b = FMIntentActionBuilder(num_input_images=3, image_sizes={"a": (256,256)}, high_builder=None)
    assert "intent" not in b.template and "action_action" not in b.template
    assert b.can_handle({"memlite_branch": "low", "intent": "grasp radio"})
    assert not b.can_handle({"memlite_branch": "high", "intent": "grasp radio"})
    assert not b.can_handle({"memlite_branch": "low", "intent": "Task complete"})
    out = {}
    b._populate_extra_samples({"memlite_branch": "low", "intent": "grasp radio"}, out)
    assert out == {"memlite_branch": "low", "intent": "grasp radio"}


def test_latest_observation_does_not_change_actions_or_physical_values():
    source = {"images": {"cam": torch.arange(6*3*2*2).reshape(6,3,2,2)},
              "state": {"arm": torch.arange(6*7).reshape(6,7)},
              "action": {"arm": torch.ones(32,7)}, "state_is_pad": torch.zeros(6).bool()}
    reduced = latest_observation(source)
    assert reduced["images"]["cam"].shape[0] == reduced["state"]["arm"].shape[0] == 1
    assert torch.equal(reduced["state"]["arm"], source["state"]["arm"][-1:])
    assert torch.equal(reduced["action"]["arm"], source["action"]["arm"])
    reduced["state"]["arm"].zero_()
    assert source["state"]["arm"][-1].sum() > 0


def test_high_low_delegate_to_different_objects_with_different_history_lengths():
    seen = {}
    high = SimpleNamespace(policy=SimpleNamespace(predict_cot=True), processor=object())
    low = SimpleNamespace(policy=SimpleNamespace(continuous_action=True, discrete_action=False))
    def high_call(obs, memories, **kwargs):
        seen["high_frames"] = obs[0]["images"]["cam"].shape[0]
        return ["plan"]
    def low_call(obs, intents):
        seen["low_frames"] = obs[0]["images"]["cam"].shape[0]
        return ["action"]
    high.infer_high_level, low.infer_low_level_action = high_call, low_call
    combined = SeparatedMEMLiteFMInferencer(high=high, low=low)
    obs = [{"images": {"cam": torch.zeros(6,3,2,2)}, "state": {"q": torch.zeros(6,27)}}]
    assert combined.infer_high_level(obs, ["memory"]) == ["plan"]
    assert combined.infer_low_level_action(obs, ["intent"]) == ["action"]
    assert seen == {"high_frames": 6, "low_frames": 1}
    with pytest.raises(RuntimeError, match="fall back"):
        combined.infer(obs)
    with pytest.raises(ValueError, match="independent"):
        SeparatedMEMLiteFMInferencer(high=high, low=high)


def test_fm_sampling_low_only_task_equal_rank_disjoint_and_resumable():
    from collections import Counter
    v = View()
    cfg = {"strategy": "fm_motion", "sampling_index_path": v.index(),
           "low_quotas": {"yaw_stop": 1, "yaw_start_reverse": 1, "gripper": 1, "recovery": 1, "other": 4}}
    seen, counts = [], Counter()
    for rank in range(4):
        s = build_memlite_train_sampler(v, sampling_config=cfg,
            batch_size=8, num_replicas=4, rank=rank, seed=17, high_fraction=0.)
        assert s.high_per_batch == 0
        batches = list(itertools.islice(s, 100))
        for batch in batches:
            assert Counter(v.lookup[i][1] for i in batch) == cfg["low_quotas"]
            counts.update(v.lookup[i][0] for i in batch)
        seen.append({i for batch in batches for i in batch})
        s.set_start_batch(19)
        assert list(itertools.islice(s, 81)) == batches[19:]
    assert len(set(counts.values())) == 1
    for rank in range(4):
        assert all(seen[rank].isdisjoint(seen[other]) for other in range(rank))


def test_fm_task_composition_preserves_baseline_critical_flags(monkeypatch):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    monkeypatch.setenv("G05_OUTPUT_DIR", "/tmp/g05-test-output")
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/"configs")):
        cfg = compose(config_name="train", overrides=["task=r1pro_memlite_fm_v11"])
    assert cfg.model.model_arch.vision.temporal_freq == 0
    assert cfg.model.model_arch.input_preprocessor.pred_eov is False
    assert cfg.model.model_arch.discrete_action is False
    assert cfg.data.obs_size == cfg.model.processor.num_obs_steps == 1
    builder = instantiate(cfg.model.processor.samples_builder)(num_input_images=3, image_sizes={"a": (256,256)})
    assert isinstance(builder, FMIntentActionBuilder)
    assert cfg.data.memlite_branch_sampling.high_fraction == 0
    assert cfg.model.max_steps == 5000 and cfg.memlite_runtime is None

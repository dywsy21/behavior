"""CPU route/freeze/cache contracts. Not a neural-model or learning gate."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import g05.models.g05.g05_policy_memlite_action as module
from g05.models.kv_cache import SparseKVCache
from g05.utils.training.ar_training_methods import ActionTrainingSettings


def bare_policy(route):
    policy = module.G05PolicyMEMLiteAction.__new__(module.G05PolicyMEMLiteAction)
    nn.Module.__init__(policy)
    policy.action_training = ActionTrainingSettings(route=route)
    policy.model = nn.Module()
    policy.model.vlm = nn.Module()
    policy.model.vlm.base = nn.Linear(2, 2)
    policy.model.vlm.lora_A = nn.Linear(2, 1, bias=False)
    policy.model.vlm.lora_B = nn.Linear(1, 2, bias=False)
    policy.model.action_expert = nn.Linear(2, 2)
    policy.model.vision_tower = nn.Linear(2, 2)
    policy.model.fm_helper = SimpleNamespace(joint_training=route == "joint")
    policy.component, policy.pipeline_stage = "low", "A"
    policy.trainability_profile = "low_ar_lora" if route == "ar" else "low_ce_fm_lora"
    policy._coordination_post_load_receipt = {"test_fixture_only": True}
    return policy


@pytest.mark.parametrize("route", ["ar", "joint", "ki"])
def test_trainability_is_route_specific(route):
    policy = bare_policy(route)
    receipt = policy.configure_coordination_trainability()
    expected = {"vlm_lora"} | ({"action_expert"} if route != "ar" else set())
    assert set(receipt["expected_trainable_groups"]) == expected
    for name, parameter in policy.named_parameters():
        assert parameter.requires_grad == ("lora_" in name or route != "ar" and "action_expert" in name)
    policy.model.vision_tower.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="parameter groups"):
        policy._assert_action_training_contract()


def test_native_task_actor_does_not_require_or_read_a_planner_projection(monkeypatch):
    from g05.data_processor.processor.samples_builder import BaseSamplesBuilder
    policy = bare_policy("ar")
    policy.action_training = ActionTrainingSettings(conditioning="native_task")
    policy.model_config = SimpleNamespace(num_input_images=1)
    mask = torch.zeros(1, 27, dtype=torch.bool)
    mask[:, [7, 8, 17, 18]] = True
    sample = dict(template=BaseSamplesBuilder(1, {"head": (256, 256)}).template,
        command="turn on the radio", embodiment="r1pro", image0={"value": torch.ones(1)},
        proprio=dict(value=torch.zeros(6, 27), proprio_dim_is_pad=mask[0]))
    def forbidden(*args):
        raise AssertionError("Native task actor must not request a semantic projection")
    monkeypatch.setattr(module, "validate_embedded_model_projection", forbidden)
    captured = []
    def prefill(samples, pixels):
        captured.extend(samples)
        raise RuntimeError("test reached target-free prefill")
    policy.prefill = prefill
    with pytest.raises(RuntimeError, match="reached target-free"):
        policy.forward_inference([sample], {}, action_dim_is_pad=mask)
    assert captured and "memlite_branch" not in captured[0]


def test_load_before_freeze_and_actual_ki_detach_setting_required():
    policy = bare_policy("ki")
    policy._coordination_post_load_receipt = None
    with pytest.raises(RuntimeError, match="Restore"):
        policy.configure_coordination_trainability()
    policy._coordination_post_load_receipt = {}
    policy.model.fm_helper.joint_training = True
    with pytest.raises(RuntimeError, match="isolation"):
        policy.configure_coordination_trainability()


def bare_model():
    model = module.ActionTrainingModelQwen35.__new__(module.ActionTrainingModelQwen35)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(ae_vlm_condition_mode="both")
    return model


def cache():
    value = SparseKVCache(last_linear_layer=1)
    value.key_cache[3] = torch.ones(1, 2, 10, 3, requires_grad=True)
    value.value_cache[3] = value.key_cache[3] * 2
    value.recurrent_states[1] = torch.full((1, 2, 3), 99.)
    return value


def test_action_suffix_requires_real_prefix_recurrent_boundary():
    model, value = bare_model(), cache()
    with pytest.raises(RuntimeError, match="prefix-boundary"):
        model._build_prefix_action_kv(value, 6)
    value.split_recurrent_states[1] = torch.ones(1, 2, 3, requires_grad=True)
    prefix = model._build_prefix_action_kv(value, 6)
    assert prefix.num_items() == 6
    assert prefix.recurrent_states[1] is value.split_recurrent_states[1]
    assert not torch.equal(prefix.recurrent_states[1], value.recurrent_states[1])
    assert not prefix.detach().recurrent_states[1].requires_grad
    value.split_recurrent_states[1] = None
    with pytest.raises(RuntimeError, match="Invalid prefix"):
        model._build_prefix_action_kv(value, 6)


def test_full_observation_prefix_inference_does_not_need_suffix_boundary():
    model, value = bare_model(), cache()
    prefix = model._build_prefix_action_kv(value, 10)
    assert prefix.recurrent_states[1] is value.recurrent_states[1]


def test_boundary_scope_restores_even_after_failed_forward(monkeypatch):
    model = bare_model()
    def failing(self, **kwargs):
        assert self._require_action_suffix_boundary
        raise RuntimeError("test forward failure")
    monkeypatch.setattr(module.G05ModelQwen35, "forward", failing)
    with pytest.raises(RuntimeError, match="test forward"):
        model(torch.ones(1, 10), continuous_action=True, split_index=6)
    assert not hasattr(model, "_action_suffix_boundary_scope")
    assert not hasattr(model, "_require_action_suffix_boundary")


def actor_fixture(route, monkeypatch):
    policy = bare_policy(route)
    policy.action_tokenizer = SimpleNamespace(action_token_begin_idx=100, action_token_end_idx=200,
        _codebook_size=10, serializer=SimpleNamespace(nn_key_names=["test"], rule_key_names=[],
            num_residuals=1, max_residuals=1, code_len=2, group_marker_action_indices={"<test>": 10}))
    state = SimpleNamespace(attention_mask=torch.ones(1, 6), pixel_values={}, kv_cache=object(),
                            position_ids=torch.arange(6)[None])
    monkeypatch.setattr(module, "validate_embedded_model_projection", lambda sample:
        dict(memlite_branch="low", task_complete=False, next_decision="EXECUTE", active_skills=[{"verb": "GRASP"}]))
    monkeypatch.setattr(module, "action_prefix_samples", lambda samples, settings: samples)
    calls = []
    def prefill(samples, images):
        assert not any("action" in sample for sample in samples)
        calls.append("prefill")
        return state
    policy.prefill = prefill
    def ar(*args, **kwargs):
        assert kwargs["only_ar"] is True and "action_gt" not in kwargs
        calls.append("ar")
        return dict(action=torch.ones(1, 32, 27), selected_action_source="ar",
                    ar_absent_keys=[set()], decoded_action_tokens=[torch.tensor([110, 101, 102])])
    policy.generate_action = ar
    def fm(**kwargs):
        calls.append("fm")
        return torch.ones(1, 32, 27)
    policy.model.inference_fm = fm
    mask = torch.zeros(1, 27, dtype=torch.bool)
    mask[:, [7, 8, 17, 18]] = True
    return policy, [{}], mask, calls


@pytest.mark.parametrize("route", ["ar", "joint", "ki"])
def test_actor_selects_one_route_and_preserves_all_real_controls(route, monkeypatch):
    policy, samples, mask, calls = actor_fixture(route, monkeypatch)
    result = policy.forward_inference(samples, {}, action_dim_is_pad=mask)
    assert calls == ["prefill", "ar" if route == "ar" else "fm"]
    assert result["execution_start"] == 0 and result["execution_steps"] == 16
    assert (result["action"][..., ~mask[0]] == 1).all()
    assert (result["action"][..., mask[0]] == 0).all()


@pytest.mark.parametrize("invalid", ["missing_group", "empty_decode", "fm_fallback"])
def test_ar_never_accepts_empty_decoder_zeros_or_fallback(invalid, monkeypatch):
    policy, samples, mask, _ = actor_fixture("ar", monkeypatch)
    result = dict(action=torch.zeros(1, 32, 27), selected_action_source="ar",
                  ar_absent_keys=[set()], decoded_action_tokens=[torch.tensor([101])])
    if invalid == "missing_group":
        result["ar_absent_keys"] = [{"lower_body"}]
    elif invalid == "empty_decode":
        result["decoded_action_tokens"] = [torch.tensor([0])]
    else:
        result["selected_action_source"] = "fm"
    policy.generate_action = lambda *args, **kwargs: result
    with pytest.raises(RuntimeError):
        policy.forward_inference(samples, {}, action_dim_is_pad=mask)


def test_actor_refuses_expert_actions_before_prefill(monkeypatch):
    policy, samples, mask, calls = actor_fixture("ar", monkeypatch)
    with pytest.raises(ValueError, match="target-free"):
        policy.forward_inference(samples, {}, actions=torch.zeros(1, 32, 27), action_dim_is_pad=mask)
    assert not calls


def test_cot_stop_is_committed_once_before_action_hidden_is_used():
    cache = SparseKVCache(last_linear_layer=1)
    cache.key_cache[3] = torch.ones(1, 1, 3, 2)
    cache.value_cache[3] = cache.key_cache[3].clone()
    forwarded = []
    class VLM:
        def embed(self, token):
            forwarded.append(token.clone())
            return token[..., None].float().repeat(1, 1, 2)
        def __call__(self, **kwargs):
            assert kwargs["kv_cache"] is cache
            cache.key_cache[3] = torch.cat([cache.key_cache[3], torch.zeros(1, 1, 1, 2)], dim=2)
            cache.value_cache[3] = cache.key_cache[3].clone()
            return torch.full((1, 1, 2), 42.), cache
    core = SimpleNamespace(vlm=VLM(), ar_helper=SimpleNamespace(_assign_token_index=lambda ids: torch.full_like(ids, 6)),
        attn_implementation="test", build_causal_mask_and_position_ids=lambda ids, mask, **kw:
            (torch.zeros(1, 1, 1, 4), torch.full((3, 1, 1), 3)),
        mask_helper=SimpleNamespace(_build_mrope_position_ids=lambda mask, device:
            torch.arange(mask.shape[1])[None, None].repeat(3, 1, 1)))
    policy = SimpleNamespace(model=core, processor=SimpleNamespace(eov_token_id=9))
    state = SimpleNamespace(generated_ids=torch.tensor([[7, 8, 9]]), kv_cache=cache,
        attention_mask=torch.ones(1, 3), position_ids=torch.arange(3)[None], last_hidden=torch.zeros(1, 2),
        device=torch.device("cpu"))
    def invariants(where):
        assert state.kv_cache.num_items() == state.attention_mask.shape[1] == state.position_ids.shape[-1]
    state.check_invariants = invariants
    returned = module.G05PolicyMEMLiteAction._commit_generated_cot_eov(policy, state)
    assert returned is state and len(forwarded) == 1 and forwarded[0].tolist() == [[9]]
    assert state.last_hidden.eq(42).all() and state.position_ids.shape == (3, 1, 4)
    with pytest.raises(RuntimeError, match="real EOV"):
        module.G05PolicyMEMLiteAction._commit_generated_cot_eov(policy, state)
    assert len(forwarded) == 1
    state.generated_ids = torch.tensor([[7, 8]])
    with pytest.raises(RuntimeError, match="real EOV"):
        module.G05PolicyMEMLiteAction._commit_generated_cot_eov(policy, state)

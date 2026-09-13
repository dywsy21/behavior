"""Small CPU states test complete native loading and zero-function adapters."""
from types import SimpleNamespace

import pytest
import torch

from native_action_initialization import verify_native_parent


def fixture():
    base = {f"model.vlm.p{i}": torch.tensor([float(i)]) for i in range(946)}
    actual = {name.replace("model.vlm.", "model.vlm.base_model.model."): value.clone()
              for name, value in base.items()}
    for i in range(96):
        actual[f"model.vlm.base_model.model.q{i}.lora_A.skill_fm.weight"] = torch.ones(1)
        actual[f"model.vlm.base_model.model.q{i}.lora_B.skill_fm.weight"] = torch.zeros(1)
    model = SimpleNamespace(state_dict=lambda: actual, _coordination_post_load_receipt=dict(
        adapter_load_mode="base_init", expected=192, restored=0, checkpoint_has_adapter=0))
    model.processor = SimpleNamespace(hl_end_token_id=None, state_token_id=252188, eov_token_id=252187,
        action_tokenizer=SimpleNamespace(action_token_begin_idx=248077, action_token_end_idx=252187))
    return model, {"model_state_dict": base}, actual


def test_native_base_is_exact_with_only_zero_function_fresh_lora():
    model, parent, _ = fixture()
    result = verify_native_parent(model, parent)
    assert result["exact_base_entries"] == 946 and result["new_lora_entries"] == 192
    assert result["zero_function_adapter"] and not result["trained_adapter_restored"]


@pytest.mark.parametrize("damage", ["missing", "base_value", "dtype", "nan", "active_adapter", "adapter_missing", "resume", "shifted_state"])
def test_random_partial_or_a4_adapter_initialization_is_rejected(damage):
    model, parent, actual = fixture()
    key = "model.vlm.base_model.model.p0"
    if damage == "missing":
        del actual[key]
    elif damage == "base_value":
        actual[key].add_(1)
    elif damage == "dtype":
        actual[key] = actual[key].half()
    elif damage == "nan":
        actual[key].fill_(float("nan"))
    elif damage == "active_adapter":
        actual["model.vlm.base_model.model.q0.lora_B.skill_fm.weight"].add_(1)
    elif damage == "adapter_missing":
        del actual["model.vlm.base_model.model.q0.lora_A.skill_fm.weight"]
    elif damage == "resume":
        model._coordination_post_load_receipt["adapter_load_mode"] = "resume"
    else:
        model.processor.state_token_id = 252189
    with pytest.raises(RuntimeError):
        verify_native_parent(model, parent)


def test_optional_boundary_preserves_legacy_layout_and_native_layout_explicitly():
    from g05.models.g05.io.input_preprocessor import register_memlite_high_end
    for config, expected in [({}, 252188), ({"register_memlite_hl_end": True}, 252188),
                             ({"register_memlite_hl_end": False}, None)]:
        added = []
        registry = SimpleNamespace(register=lambda names: added.extend(names), get_id=lambda name: 252188)
        assert register_memlite_high_end(registry, config) == expected
        assert added == ([] if expected is None else ["<HL_END>"])
    with pytest.raises(ValueError):
        register_memlite_high_end(registry, {"register_memlite_hl_end": "false"})
    with pytest.raises(ValueError):
        register_memlite_high_end(registry, {"register_memlite_hl_end": False, "memlite_train_mode": "high"})


def test_policy_constructor_uses_the_declared_processor_extension():
    from g05.models.g05 import g05_policy
    from g05.models.g05.io.input_preprocessor import InputPreprocessor
    assert g05_policy.InputPreprocessor is InputPreprocessor

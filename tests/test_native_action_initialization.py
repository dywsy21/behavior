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
    return model, {"model_state_dict": base}, actual


def test_native_base_is_exact_with_only_zero_function_fresh_lora():
    model, parent, _ = fixture()
    result = verify_native_parent(model, parent)
    assert result["exact_base_entries"] == 946 and result["new_lora_entries"] == 192
    assert result["zero_function_adapter"] and not result["trained_adapter_restored"]


@pytest.mark.parametrize("damage", ["missing", "base_value", "dtype", "nan", "active_adapter", "adapter_missing", "resume"])
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
    else:
        model._coordination_post_load_receipt["adapter_load_mode"] = "resume"
    with pytest.raises(RuntimeError):
        verify_native_parent(model, parent)

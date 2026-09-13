"""Identify the released G0.5 base and verify a zero-function LoRA warm start.

This is a native checkpoint initialization, not the published BEHAVIOR policy
or a reproduction of its full training recipe. It never borrows A4 adapters.
"""
from pathlib import Path

NATIVE_PARENT = Path("/mnt/sdc1/robodojo/checkpoints/G05/g05-base/checkpoints/model_state_dict.pt")
NATIVE_PARENT_SHA = "072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327"
NATIVE_CONFIG = NATIVE_PARENT.parent.parent / ".hydra/config.yaml"
NATIVE_CONFIG_SHA = "c98af37352c0f600341d2ecbdb49fcdfa87812198654448991615065fa1a4461"
HF_DOWNLOAD_REVISION = "e312be81e90c56a55bcb26b57429bd39a335b449"


def verify_native_parent(model, parent):
    import torch
    from g05.models.g05.helpers.vlm_lora import normalize_pre_injection_state_keys

    if set(parent) != {"model_state_dict"}:
        raise RuntimeError("Native released checkpoint must contain weights only")
    expected = parent["model_state_dict"]
    actual = model.state_dict()
    adapters = {name: value for name, value in actual.items() if "lora_" in name}
    base = normalize_pre_injection_state_keys({name: value for name, value in actual.items() if name not in adapters})
    if (len(expected) != 946 or set(base) != set(expected)
            or any("lora_" in name for name in expected)):
        raise RuntimeError("Native base restoration must cover exactly all 946 released states")
    for name, value in base.items():
        left, right = value.detach().cpu(), expected[name].detach().cpu()
        if (left.shape != right.shape or left.dtype != right.dtype or not torch.isfinite(left).all()
                or not torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                   right.contiguous().reshape(-1).view(torch.uint8))):
            raise RuntimeError("Native base tensor was not restored exactly: " + name)
    a = {name: value for name, value in adapters.items() if ".lora_A." in name}
    b = {name: value for name, value in adapters.items() if ".lora_B." in name}
    if (len(adapters) != 192 or len(a) != 96 or len(b) != 96
            or any(not name.startswith("model.vlm.") for name in adapters)
            or any(not torch.isfinite(value).all() for value in adapters.values())
            or any(torch.count_nonzero(value) for value in b.values())):
        raise RuntimeError("Native initialization requires 192 finite fresh LoRA states and all-zero B")
    receipt = getattr(model, "_coordination_post_load_receipt", {})
    if (receipt.get("adapter_load_mode") != "base_init" or receipt.get("expected") != 192
            or receipt.get("restored") != 0 or receipt.get("checkpoint_has_adapter") != 0):
        raise RuntimeError("Native initialization must not restore an A4/trained adapter")
    return dict(passed=True, exact_base_entries=946, new_lora_entries=192,
                zero_function_adapter=True, trained_adapter_restored=False,
                published_behavior_finetune_replication=False)

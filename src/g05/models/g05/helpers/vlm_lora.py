"""Controlled PEFT LoRA lifecycle for the low-level VLM.

PEFT changes state-dict paths below the wrapped module from ``layers.*`` to
``base_model.model.layers.*``.  The policy uses these helpers only *after* a
base checkpoint is loaded.  The checkpoint loader owns calling the remap and
post-load hooks before DDP/optimizer construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


DEFAULT_VLM_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class VLMloraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_VLM_LORA_TARGETS
    adapter_name: str = "skill_fm"

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "VLMloraConfig":
        value = dict(value or {})
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("low_vlm_lora.enabled must be a boolean, not a truthy string/value")
        result = cls(
            enabled=enabled,
            r=int(value.get("r", 8)),
            alpha=int(value.get("alpha", value.get("lora_alpha", 16))),
            dropout=float(value.get("dropout", value.get("lora_dropout", 0.0))),
            target_modules=tuple(value.get("target_modules", DEFAULT_VLM_LORA_TARGETS)),
            adapter_name=str(value.get("adapter_name", "skill_fm")),
        )
        if result.r < 1 or result.alpha < 1:
            raise ValueError("LoRA r and alpha must be positive")
        if not 0.0 <= result.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not result.target_modules or any(not str(name) for name in result.target_modules):
            raise ValueError("LoRA target_modules must be nonempty strings")
        if not result.adapter_name:
            raise ValueError("LoRA adapter_name must be nonempty")
        return result


def is_peft_wrapped(module: object) -> bool:
    return type(module).__name__.startswith("PeftModel")


def inject_vlm_lora(vlm, config: VLMloraConfig):
    """Wrap one VLM after its base weights have been restored."""
    if not config.enabled:
        return vlm
    if is_peft_wrapped(vlm):
        return vlm
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RuntimeError("low_vlm_lora.enabled=true requires the peft package") from error
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        # MixtureQwen35 consumes embeddings and our SparseKVCache directly; it
        # is not a Hugging Face feature-extraction forward.  The specialized
        # PEFT wrapper injects input_ids / return_dict / output_hidden_states
        # and fails before the FM graph runs.  The generic wrapper forwards
        # the original arguments without changing cache or output semantics.
        task_type=None,
    )
    wrapped = get_peft_model(vlm, peft_config, adapter_name=config.adapter_name)
    if not any("lora_" in name and parameter.requires_grad
               for name, parameter in wrapped.named_parameters()):
        raise RuntimeError("PEFT injection produced no trainable LoRA parameters")
    return wrapped


def normalize_pre_injection_state_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Alias PEFT base keys back to the unwrapped G05 VLM before base loading.

    Adapter keys intentionally remain in the dict.  The normal base loader may
    report them as unexpected; ``restore_lora_state`` consumes them after
    injection.  A collision signals an ambiguous checkpoint rather than silently
    choosing one tensor.
    """
    prefix = "model.vlm.base_model.model."
    output: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = "model.vlm." + key[len(prefix):] if key.startswith(prefix) else key
        if key.startswith(prefix) and "lora_" not in key:
            # PEFT's Linear stores its original tensor below base_layer.
            # The base checkpoint loader runs before PEFT is injected, so it
            # must see q_proj.weight, not q_proj.base_layer.weight. Restoring
            # only the adapters would otherwise leave base weights missing.
            normalized = normalized.replace(".base_layer.", ".")
        if normalized in output:
            raise ValueError(f"Ambiguous LoRA/base checkpoint alias for {normalized}")
        output[normalized] = value
    return output


def restore_lora_state(policy, state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Restore only LoRA tensors after :func:`inject_vlm_lora`.

    Returns coverage counts and rejects a partial adapter load.  The policy's
    full state-dict prefix is used so this works without the caller stripping
    ``model.`` by hand.
    """
    expected = {
        name for name, _ in policy.named_parameters() if "model.vlm." in name and "lora_" in name
    }
    all_adapter_keys = {name for name in state_dict if "lora_" in name}
    available = {name: value for name, value in state_dict.items() if "model.vlm." in name and "lora_" in name}
    if all_adapter_keys and not available:
        raise RuntimeError(
            "checkpoint claims a LoRA adapter but none use the expected model.vlm namespace; "
            "refusing random adapter initialization on resume"
        )
    if not available:
        return {
            "expected": len(expected), "restored": 0, "checkpoint_has_adapter": 0,
            "adapter_load_mode": "base_init",
        }
    message = policy.load_state_dict(available, strict=False)
    restored = expected - set(message.missing_keys)
    unexpected = [key for key in message.unexpected_keys if "lora_" in key]
    if unexpected or restored != expected:
        raise RuntimeError(
            "LoRA adapter restore coverage failed: "
            f"expected={len(expected)} restored={len(restored)} unexpected={unexpected[:3]}"
        )
    return {
        "expected": len(expected), "restored": len(restored), "checkpoint_has_adapter": len(available),
        "adapter_load_mode": "resume",
    }

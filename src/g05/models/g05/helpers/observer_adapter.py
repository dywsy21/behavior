"""A result-only LoRA branch, separate from planner and action training."""


def require_observer_adapter_only(policy):
    names=[name for name,p in policy.named_parameters() if p.requires_grad]
    if not names or any(not name.startswith('model.vlm.')
                        or '.lora_' not in name or '.outcome_observer.' not in name for name in names):
        raise ValueError('Only dedicated outcome_observer VLM LoRA may receive prefix gradients')
    return names

"""Small, explicit helpers for excluding disabled model branches from training.

The helpers deliberately work by parameter *name* rather than by a model-class
check.  This lets a task configuration declare a frozen branch while keeping
the base model usable for its other training modes.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn


def freeze_parameters_by_prefixes(
    module: nn.Module, prefixes: Iterable[str]
) -> dict[str, Any]:
    """Freeze every parameter whose fully qualified name starts with a prefix.

    Empty prefix lists are a no-op. A non-empty prefix list is fail-fast: each
    requested prefix has to match at least one parameter, avoiding a silent
    typo that would leave a disabled branch inside DDP/AdamW.
    """
    normalized = tuple(str(prefix) for prefix in prefixes if str(prefix))
    named_parameters = list(module.named_parameters())
    missing = [
        prefix
        for prefix in normalized
        if not any(name.startswith(prefix) for name, _ in named_parameters)
    ]
    if missing:
        raise ValueError(
            "freeze_parameter_prefixes matched no parameters for: "
            + ", ".join(repr(prefix) for prefix in missing)
        )

    selected = [
        (name, parameter)
        for name, parameter in named_parameters
        if any(name.startswith(prefix) for prefix in normalized)
    ]
    already_frozen = sum(not parameter.requires_grad for _, parameter in selected)
    for _, parameter in selected:
        parameter.requires_grad_(False)

    trainable = [parameter for _, parameter in module.named_parameters() if parameter.requires_grad]
    return {
        "requested_prefixes": list(normalized),
        "frozen_parameter_names": [name for name, _ in selected],
        "frozen_parameter_tensors": len(selected),
        "frozen_parameter_numel": sum(parameter.numel() for _, parameter in selected),
        "already_frozen_parameter_tensors": already_frozen,
        "total_parameter_tensors": len(named_parameters),
        "total_parameter_numel": sum(parameter.numel() for _, parameter in named_parameters),
        "trainable_parameter_tensors_after_freeze": len(trainable),
        "trainable_parameter_numel_after_freeze": sum(parameter.numel() for parameter in trainable),
    }

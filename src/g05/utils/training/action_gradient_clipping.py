"""Explicit low-level optimizer clipping; never installed automatically.

Call after accumulation and DDP gradient reduction (and AMP unscaling, if used),
before optimizer.step(). This partitions PARAMETERS, not action dimensions or
individual losses. Per-module clipping does not measure/project task conflicts.
The original global path retains the caller's parameter order and PyTorch op.
"""
from __future__ import annotations

import math

import torch


def _validate(named_parameters, parameter_groups, mode, max_norm, foreach):
    if mode not in {"global", "per_module"}:
        raise ValueError("Select explicit global or per_module clipping")
    if (type(max_norm) not in {int, float} or not math.isfinite(max_norm) or max_norm <= 0
            or foreach is not None and type(foreach) is not bool):
        raise ValueError("Clipping requires a positive finite max_norm and boolean/None foreach")
    if not isinstance(parameter_groups, dict) or set(parameter_groups) not in (
            {"vlm_lora"}, {"vlm_lora", "action_expert"}):
        raise ValueError("Only declared low-level LoRA / optional action-expert groups are supported")
    ordered, by_name, ids = [], {}, set()
    for name, parameter in named_parameters:
        if (not isinstance(name, str) or not name or name in by_name
                or not isinstance(parameter, torch.nn.Parameter) or id(parameter) in ids):
            raise ValueError("Model parameters must have unique names and tensor identities")
        by_name[name] = parameter
        ids.add(id(parameter))
        if parameter.requires_grad:
            ordered.append(parameter)
        elif parameter.grad is not None:
            raise ValueError("Frozen parameters must not retain gradients")
    grouped, seen = {}, set()
    for group, members in parameter_groups.items():
        members = list(members)
        if not members:
            raise ValueError("Declared parameter groups must not be empty")
        grouped[group] = []
        for name, parameter in members:
            if (name not in by_name or by_name[name] is not parameter
                    or not parameter.requires_grad or id(parameter) in seen):
                raise ValueError("Group identity, duplicate membership or frozen parameter mismatch")
            seen.add(id(parameter))
            grouped[group].append(parameter)
    if not ordered or seen != {id(parameter) for parameter in ordered}:
        raise ValueError("Groups must cover every and only trainable parameter exactly once")
    if len({parameter.device for parameter in ordered}) != 1:
        raise ValueError("This low-level helper expects one device per DDP rank")
    for parameter in ordered:
        if parameter.grad is not None and parameter.grad.layout != torch.strided:
            raise ValueError("Only dense gradients are supported")
    return ordered, grouped


def _coefficient(norm, max_norm):
    # Same tensor dtype, epsilon and clamping as torch.nn.utils.clip_grad_norm_.
    return torch.clamp(max_norm / (norm + 1e-6), max=1.0)


@torch.no_grad()
def clip_action_gradients(named_parameters, parameter_groups, *, mode="global", max_norm=1.0,
                          foreach=None):
    """Clip with audited coverage and return pre-clip module norms/coefficients.

    `named_parameters` is the model's original ordered named_parameters(),
    including frozen tensors. Groups are its actual coordination groups of
    (name, Parameter) pairs. None gradients stay None: no optimizer states are
    fabricated for currently disconnected tensors.

    Validation and all norm checks happen before any mutation, so a missing
    group or nonfinite gradient cannot leave the other group partly clipped.
    This is not a rollback guarantee for an asynchronous device/hardware fault.

    Each group receives max_norm in per_module mode. Thus two active groups can
    have a combined norm up to sqrt(2)*max_norm; there is deliberately no second
    global clip that would reintroduce coupling. No loss weights or LRs change.
    Requires the installed PyTorch get_total_norm API (verified on 2.7.1).
    """
    ordered, groups = _validate(named_parameters, parameter_groups, mode, max_norm, foreach)
    gradients = [parameter.grad for parameter in ordered if parameter.grad is not None]
    total = torch.nn.utils.get_total_norm(gradients, norm_type=2.0,
        error_if_nonfinite=True, foreach=foreach)
    group_norms = {name: torch.nn.utils.get_total_norm(
        [parameter.grad for parameter in parameters if parameter.grad is not None],
        norm_type=2.0, error_if_nonfinite=True, foreach=foreach)
        for name, parameters in groups.items()}
    # Keep the original operation and original order for an explicit global
    # comparison; the read-only diagnostics above consume no RNG/gradient data.
    if mode == "global":
        torch.nn.utils.clip_grad_norm_(ordered, max_norm, norm_type=2.0,
            error_if_nonfinite=True, foreach=foreach)
    else:
        for parameters in groups.values():
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0,
                error_if_nonfinite=True, foreach=foreach)
    return dict(mode=mode, max_norm=float(max_norm), pre_clip_total_norm=float(total),
        groups={name: dict(parameter_tensors=len(parameters),
            gradient_tensors=sum(parameter.grad is not None for parameter in parameters),
            pre_clip_norm=float(group_norms[name]),
            applied_coefficient=float(_coefficient(total if mode == "global" else group_norms[name], max_norm)))
            for name, parameters in groups.items()},
        per_loss_gradient_measurement=False, learning_rate_multiplier_claim=False)

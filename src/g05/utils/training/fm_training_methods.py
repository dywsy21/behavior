"""Opt-in, scoped FM ablations; never change the reference evaluator or actor.

Use the context around ONE training forward. Original instance attributes are
restored even on failure. The legacy helper implementation remains the metric
reference, with the same noise/time RNG consumption when both options are off.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from types import MethodType

import torch


@dataclass(frozen=True)
class FMTrainingMethods:
    time_sampling: str = "iid"
    execution_horizon: int = 16
    execution_weight: float = 1.0

    def __post_init__(self):
        if self.time_sampling not in {"iid", "beta_stratified"}:
            raise ValueError("time_sampling must be iid or beta_stratified")
        if type(self.execution_horizon) is not int or self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be a positive integer")
        if not math.isfinite(self.execution_weight) or self.execution_weight <= 0:
            raise ValueError("execution_weight must be finite and strictly positive")

    def as_dict(self):
        return asdict(self)


def stratified_beta_times(num_samples, batch_size, *, flow_sig_min, time_convention):
    """[N,B] samples, one original Beta(1.5,1) quantile stratum per observation.

    An equal-weight mixture of the N strata has the original Beta law. Sampling
    is on CPU, as in the original helper, leaving CUDA noise RNG independent.
    """
    if type(num_samples) is not int or num_samples < 2:
        raise ValueError("stratification needs at least two flow samples")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(flow_sig_min) or not 0 <= flow_sig_min < 1:
        raise ValueError("flow_sig_min must be in [0,1)")
    if time_convention not in {"pi_convention", "galaxea_convention"}:
        raise ValueError("unknown flow time convention")
    u = (torch.arange(num_samples, dtype=torch.float32)[:, None]
         + torch.rand(num_samples, batch_size)) / num_samples
    z = u.pow(2.0 / 3.0)
    if time_convention == "pi_convention":
        return 1 - (1 - flow_sig_min) * (1 - z)
    return (1 - flow_sig_min) * (1 - z)


@contextmanager
def fm_training_methods(helper, settings: FMTrainingMethods, *, batch_size: int):
    """Temporarily replace only the explicitly selected training operations.

    No tensor/parameter/codec/normalizer changes. Do not hold this scope across
    evaluation, inference, or concurrent requests on the same helper instance.
    """
    sentinel = "_fm_training_methods_active"
    if getattr(helper, sentinel, False):
        raise RuntimeError("nested/concurrent FM method scopes are forbidden")
    if settings.execution_weight != 1 and settings.execution_horizon > helper.horizon_steps:
        raise ValueError("execution horizon exceeds predicted horizon")
    n = int(helper.num_flow_samples)
    times = None
    if settings.time_sampling == "beta_stratified":
        if helper.flow_sampling != "beta":
            raise ValueError("Beta stratification cannot silently replace a uniform objective")
        times = stratified_beta_times(n, batch_size, flow_sig_min=helper.flow_sig_min,
                                      time_convention=helper.time_convention)
    missing = object()
    saved = {name: helper.__dict__.get(name, missing)
             for name in ("sample_time", "cal_fm_loss", sentinel)}
    used = 0

    def sample_time(instance, bsz, device=None):
        nonlocal used
        if bsz != batch_size or used >= n:
            raise RuntimeError("unexpected batch/number of time draws in one FM forward")
        value = times[used]
        used += 1
        return value.to(device) if device is not None else value

    def weighted_loss(instance, v_psi, x0, x1, action_pad_masks, action_dim_is_pad=None):
        if v_psi.shape != x0.shape or v_psi.shape != x1.shape or v_psi.ndim != 3:
            raise ValueError("FM prediction/noise/target shapes differ")
        if (action_pad_masks.dtype != torch.bool
                or action_pad_masks.shape != v_psi.shape[:2]):
            raise ValueError("invalid temporal padding mask")
        target = x0 - x1 if instance.time_convention == "pi_convention" else x1 - x0
        weights = torch.ones_like(v_psi)
        weights[action_pad_masks] = instance.padding_action_weight
        if action_dim_is_pad is not None and not instance.zero_pad_action_target:
            if (action_dim_is_pad.dtype != torch.bool
                    or action_dim_is_pad.shape != (v_psi.shape[0], v_psi.shape[2])):
                raise ValueError("invalid dimension padding mask")
            weights.masked_fill_(action_dim_is_pad[:, None, :], instance.padding_action_weight)
        weights[:, :settings.execution_horizon, :] *= settings.execution_weight
        return ((v_psi - target).square() * weights).sum() / weights.sum().clamp(min=1)

    setattr(helper, sentinel, True)
    try:
        if times is not None:
            helper.sample_time = MethodType(sample_time, helper)
        if settings.execution_weight != 1:
            helper.cal_fm_loss = MethodType(weighted_loss, helper)
        yield
        if times is not None and used != n:
            raise RuntimeError("FM forward did not consume its complete time-strata set")
    finally:
        for name, value in saved.items():
            if value is missing:
                helper.__dict__.pop(name, None)
            else:
                setattr(helper, name, value)


def executed_prefix_codec_input(actions: torch.Tensor, execution_horizon: int = 16):
    """Experimental codec input: exact executable prefix, last-value tail pad.

    The suffix is a codec padding convention, NOT an expert-action label for
    later physical timesteps. Decode and execute only the declared prefix.
    """
    if actions.ndim != 3 or not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("actions must be finite floating-point [B,H,D]")
    if type(execution_horizon) is not int or not 0 < execution_horizon <= actions.shape[1]:
        raise ValueError("invalid executable horizon")
    prefix = actions[:, :execution_horizon]
    tail = prefix[:, -1:].expand(-1, actions.shape[1] - execution_horizon, -1)
    return torch.cat((prefix, tail), dim=1)

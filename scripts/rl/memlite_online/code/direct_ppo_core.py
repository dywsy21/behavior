"""Small, model-agnostic PPO core for a stochastic flow policy.

This module intentionally contains no A4 imports.  The A4 adapter supplies a
``velocity_fn(x, t)`` which calls its existing action expert.  Keeping the
probability math separate makes it possible to test the critical PPO invariant:
the policy which collected a chain must recompute a probability ratio of one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


VelocityFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class FlowChain:
    """A sampled flow trajectory and its transition log probability.

    ``states`` has shape ``[batch, flow_steps + 1, horizon, action_dim]``.
    The initial standard-normal density is deliberately omitted: it does not
    depend on trainable A4 parameters and cancels from the PPO ratio.
    """

    states: Tensor
    log_prob: Tensor
    step_log_prob: Tensor


def _validate_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.dtype is not torch.bool:
        raise TypeError("active_action_dims must be boolean")
    if mask.ndim != 1 or mask.shape[0] != reference.shape[-1]:
        raise ValueError("active_action_dims must have shape [action_dim]")
    if not bool(mask.any()):
        raise ValueError("at least one action dimension must be active")
    return mask.to(device=reference.device)


def _step_times(batch: int, flow_steps: int, *, pi_convention: bool,
                device: torch.device, dtype: torch.dtype) -> Tensor:
    if flow_steps <= 0:
        raise ValueError("flow_steps must be positive")
    if pi_convention:
        values = torch.arange(flow_steps, 0, -1, device=device, dtype=dtype) / flow_steps
    else:
        values = torch.arange(flow_steps, device=device, dtype=dtype) / flow_steps
    return values.unsqueeze(0).expand(batch, -1)


def masked_normal_log_prob(sample: Tensor, mean: Tensor, std: Tensor,
                           active_action_dims: Tensor) -> Tensor:
    """Return one joint log probability per batch item and flow step."""
    if sample.shape != mean.shape:
        raise ValueError("sample and mean shapes differ")
    if torch.any(std <= 0) or not torch.isfinite(std).all():
        raise ValueError("std must be finite and positive")
    mask = _validate_mask(active_action_dims, sample)
    elementwise = torch.distributions.Normal(mean, std).log_prob(sample)
    return elementwise[..., mask].sum(dim=(-2, -1))


@torch.no_grad()
def sample_stochastic_flow(
    velocity_fn: VelocityFn,
    initial_noise: Tensor,
    transition_std: Tensor | float,
    active_action_dims: Tensor,
    *,
    flow_steps: int,
    pi_convention: bool = True,
    generator: torch.Generator | None = None,
) -> FlowChain:
    """Sample the augmented Markov chain used for direct PPO fine-tuning.

    A4 uses the pi convention: time starts at one and each Euler step subtracts
    velocity.  Noise is added *after* each Euler mean so every transition has a
    tractable Gaussian probability.  No clipping is performed here because a
    clipped sample no longer has the unmodified Gaussian density.
    """
    if initial_noise.ndim != 3:
        raise ValueError("initial_noise must have shape [batch, horizon, action_dim]")
    if not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise contains non-finite values")
    mask = _validate_mask(active_action_dims, initial_noise)
    times = _step_times(initial_noise.shape[0], flow_steps,
                        pi_convention=pi_convention,
                        device=initial_noise.device, dtype=initial_noise.dtype)
    dt = 1.0 / flow_steps
    direction = -1.0 if pi_convention else 1.0
    std = torch.as_tensor(transition_std, device=initial_noise.device,
                          dtype=initial_noise.dtype)
    if torch.any(std <= 0) or not torch.isfinite(std).all():
        raise ValueError("transition_std must be finite and positive")

    current = initial_noise.clone()
    states = [current.clone()]
    step_log_probs = []
    for step in range(flow_steps):
        velocity = velocity_fn(current, times[:, step])
        if velocity.shape != current.shape or not torch.isfinite(velocity).all():
            raise ValueError("velocity_fn returned an invalid tensor")
        mean = current + direction * dt * velocity
        noise = torch.randn(current.shape, device=current.device, dtype=current.dtype,
                            generator=generator)
        candidate = mean + std * noise
        # Padded model dimensions remain deterministic zero and are excluded
        # from the probability.  They must never become an exploration channel.
        candidate[..., ~mask] = 0
        mean[..., ~mask] = 0
        step_log_probs.append(masked_normal_log_prob(candidate, mean, std, mask))
        current = candidate
        states.append(current.clone())
    per_step = torch.stack(step_log_probs, dim=1)
    return FlowChain(states=torch.stack(states, dim=1),
                     log_prob=per_step.sum(dim=1), step_log_prob=per_step)


def recompute_flow_step_log_probs(
    velocity_fn: VelocityFn,
    states: Tensor,
    transition_std: Tensor | float,
    active_action_dims: Tensor,
    *,
    pi_convention: bool = True,
) -> Tensor:
    """Recompute one log probability per flow transition.

    Diffusion/flow PPO treats each denoising transition as a policy step.  PPO
    clipping therefore applies to these values separately instead of to the
    product of every transition in the full generated chain.
    """
    if states.ndim != 4 or states.shape[1] < 2:
        raise ValueError("states must have shape [batch, flow_steps + 1, horizon, action_dim]")
    mask = _validate_mask(active_action_dims, states)
    flow_steps = states.shape[1] - 1
    times = _step_times(states.shape[0], flow_steps, pi_convention=pi_convention,
                        device=states.device, dtype=states.dtype)
    dt = 1.0 / flow_steps
    direction = -1.0 if pi_convention else 1.0
    std = torch.as_tensor(transition_std, device=states.device, dtype=states.dtype)
    result = []
    for step in range(flow_steps):
        previous = states[:, step]
        velocity = velocity_fn(previous, times[:, step])
        mean = previous + direction * dt * velocity
        mean = mean.clone()
        mean[..., ~mask] = 0
        result.append(masked_normal_log_prob(states[:, step + 1], mean, std, mask))
    return torch.stack(result, dim=1)


def recompute_flow_log_prob(
    velocity_fn: VelocityFn,
    states: Tensor,
    transition_std: Tensor | float,
    active_action_dims: Tensor,
    *,
    pi_convention: bool = True,
) -> Tensor:
    """Recompute transition probability with gradients for a stored chain."""
    return recompute_flow_step_log_probs(
        velocity_fn, states, transition_std, active_action_dims,
        pi_convention=pi_convention,
    ).sum(dim=1)


def ppo_clipped_policy_loss(new_log_prob: Tensor, old_log_prob: Tensor,
                            advantages: Tensor, *, clip_ratio: float) -> tuple[Tensor, dict[str, Tensor]]:
    """Standard PPO clipped objective with compact diagnostics."""
    if not 0 < clip_ratio < 1:
        raise ValueError("clip_ratio must be between zero and one")
    if new_log_prob.shape != old_log_prob.shape or new_log_prob.shape != advantages.shape:
        raise ValueError("PPO tensors must have identical shapes")
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    with torch.no_grad():
        metrics = {
            "ratio_mean": ratio.mean(),
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "clip_fraction": ((ratio - 1).abs() > clip_ratio).float().mean(),
        }
    return loss, metrics

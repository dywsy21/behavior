"""H0 observer adapter over <=4 CAUSAL frozen VLM prefills, not AR answers.

Deploy through a separately calibrated/readiness-gated observer. Adding this
module alone must not enable it in the stage-1 planner or training actor.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .outcome_head import PlannerOutcomeHead


class TemporalOutcomeObserver(nn.Module):
    """Each batch row conditions on ONE named skill member in its prefix.

    Parallel members require separate conditioned contexts; duplicating a
    shared full-bundle vector cannot produce identifiable per-member labels.
    The feature-extraction caller must preserve that binding and its SHA.
    """
    def __init__(self, hidden_size, width=128, *, include_absolute_proprio=False,include_served_controls=False):
        super().__init__()
        if type(include_absolute_proprio) is not bool:
            raise ValueError('Explicit observer proprio architecture flag required')
        self.include_absolute_proprio=include_absolute_proprio
        if type(include_served_controls) is not bool:
            raise ValueError('Explicit observable command-age architecture flag required')
        self.include_served_controls=include_served_controls
        self.context_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.SiLU())
        self.sequence = nn.GRU(width+28+(27 if include_absolute_proprio else 0)+int(include_served_controls), width, batch_first=True)
        self.residual = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.head = PlannerOutcomeHead(hidden_size)

    def forward(self, context, proprio, steps, valid, *, allow_context_grad=False,served_controls=None):
        if type(allow_context_grad) is not bool:
            raise ValueError('Explicit context-gradient opt-in required')
        if allow_context_grad and torch.is_grad_enabled() and not self.sequence.training:
            raise ValueError('Observer adapter backward requires GRU training mode')
        if context.ndim != 3 or not 1 <= context.shape[1] <= 4:
            raise ValueError("Observer admits 1..4 past/current checkpoints")
        batch, length, _ = context.shape
        if proprio.shape != (batch, length, 27) or steps.shape != (batch, length) or valid.shape != steps.shape:
            raise ValueError("Unaligned causal observer batch")
        if valid.dtype != torch.bool or steps.dtype not in (torch.int32, torch.int64):
            raise ValueError("Strict temporal mask and real integer control clocks required")
        sizes = valid.sum(1)
        wanted = torch.arange(length, device=valid.device)[None] < sizes[:, None]
        if (not sizes.gt(0).all() or not torch.equal(valid, wanted)
                or not torch.isfinite(context).all() or not torch.isfinite(proprio).all()
                or (steps[valid] < 0).any()
                or ((steps[:, 1:] <= steps[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("Temporal observations must be finite, right-padded, and strictly causal")
        if self.include_served_controls:
            if (served_controls is None or served_controls.shape!=steps.shape
                    or served_controls.dtype not in (torch.int32,torch.int64)
                    or (served_controls[valid]<0).any() or (served_controls[valid]>steps[valid]).any()):
                raise ValueError('Need the actual nonnegative served-control count at each causal check')
            starts=steps-served_controls
            if ((starts!=starts[:,:1])&valid).any():
                raise ValueError('Observer sequence crosses an issued intent attempt')
        elif served_controls is not None:
            raise ValueError('Served-control input requires its matching observer architecture')
        # Cached H0 remains insulated by default. A separate observer-only
        # adapter experiment can opt in; that caller must freeze the planner,
        # visual backbone and actor and verify its dedicated LoRA boundary.
        feature = (context if allow_context_grad else context.detach()).float()
        state = proprio.detach().float()
        delta = torch.zeros_like(state)
        delta[:, 1:] = state[:, 1:] - state[:, :-1]
        dt = torch.zeros_like(steps, dtype=torch.float32)
        dt[:, 1:] = (steps[:, 1:] - steps[:, :-1]).float().clamp_min(0) / 30.
        pieces=[self.context_projection(feature),delta.tanh(),dt.log1p()[...,None]]
        if self.include_absolute_proprio:
            # These are existing normalized observable robot joints, not
            # object/contact truth. In particular absolute gripper aperture
            # must not exist only as rounded tokens inside a frozen VLM.
            pieces.append(state.tanh())
        if self.include_served_controls:
            # Unlike inter-frame dt, age distinguishes an unexecuted new
            # intent (0) from a short attempt with one available observation.
            # This is an observed command counter, never a physical label.
            pieces.append((served_controls.detach().float()/30.).log1p()[...,None])
        inputs = torch.cat(pieces, dim=-1)
        packed = pack_padded_sequence(inputs, sizes.cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.sequence(packed)
        current = feature[torch.arange(batch, device=feature.device), sizes-1]
        return self.head(current + self.residual(last[0]))

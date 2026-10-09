"""H0 observer adapter over <=4 CAUSAL frozen VLM prefills, not AR answers.

Deploy through a separately calibrated/readiness-gated observer. Adding this
module alone must not enable it in the stage-1 planner or training actor.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .outcome_head import PlannerOutcomeHead


class TemporalOutcomeObserver(nn.Module):
    def __init__(self, hidden_size, width=128):
        super().__init__()
        self.context_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.SiLU())
        self.sequence = nn.GRU(width+28, width, batch_first=True)
        self.residual = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.head = PlannerOutcomeHead(hidden_size)

    def forward(self, context, proprio, steps, valid):
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
        # H0 is insulated: outcome loss cannot update high VLM or low controller.
        feature = context.detach().float()
        state = proprio.detach().float()
        delta = torch.zeros_like(state)
        delta[:, 1:] = state[:, 1:] - state[:, :-1]
        dt = torch.zeros_like(steps, dtype=torch.float32)
        dt[:, 1:] = (steps[:, 1:] - steps[:, :-1]).float().clamp_min(0) / 30.
        inputs = torch.cat((self.context_projection(feature), delta.tanh(), dt.log1p()[..., None]), dim=-1)
        packed = pack_padded_sequence(inputs, sizes.cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.sequence(packed)
        current = feature[torch.arange(batch, device=feature.device), sizes-1]
        return self.head(current + self.residual(last[0]))

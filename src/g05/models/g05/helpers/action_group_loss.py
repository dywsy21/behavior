"""Token-group CE weights; group identity never crosses a sample boundary."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ActionGroupLoss:
    code_begin: int
    code_size: int
    markers: dict[str, int]
    lower_body_weight: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.lower_body_weight) or self.lower_body_weight <= 0:
            raise ValueError("lower_body_weight must be finite and positive")
        if self.code_size <= 0 or len(set(self.markers.values())) != len(self.markers):
            raise ValueError("Invalid grouped action vocabulary")
        if not any(k.startswith("<lower_body_") for k in self.markers):
            raise ValueError("Lower-body supervision requires explicit group markers")

    def masks(self, labels):
        if labels.ndim != 2:
            raise ValueError("Group classification requires [batch, sequence] labels")
        codes = (labels >= self.code_begin) & (labels < self.code_begin + self.code_size)
        # A non-code token ends the previous group (including text/EOS/padding).
        # Each explicit marker starts a new group. Codes before a marker have
        # no group, even if the preceding sample ended in lower-body codes.
        event = ~codes
        lower_event = torch.zeros_like(codes)
        for name, marker in self.markers.items():
            if name.startswith("<lower_body_"):
                lower_event |= labels == marker
        positions = torch.arange(labels.shape[1], device=labels.device).expand_as(labels)
        last = torch.where(event, positions, -1).cummax(dim=1).values
        lower = codes & (last >= 0) & lower_event.gather(1, last.clamp_min(0))
        weights = torch.where(lower, self.lower_body_weight, 1.0).float()
        return weights, lower

    def reduce(self, token_loss, labels):
        weights, lower = self.masks(labels)
        valid = labels != -100
        weights, lower = weights[valid], lower[valid]
        if token_loss.shape != weights.shape:
            raise ValueError("Loss vector and shifted valid labels disagree")
        objective = (token_loss * weights).sum() / weights.sum().clamp_min(1)
        return objective, weights, lower

"""Leak-free outcome supervision for the MEM-Lite planner.

The outcome classifier intentionally receives a representation from the
planner's *context* prefix only.  It never consumes the autoregressive answer
tokens that contain the ground-truth outcome/decision during teacher forcing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


OUTCOME_NAMES = ("IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN")
OUTCOME_TO_ID = {name: index for index, name in enumerate(OUTCOME_NAMES)}


def normalize_outcome(value: object) -> str:
    """Normalize one protocol outcome without silently inventing a label."""
    normalized = str(value or "").strip().upper()
    if normalized not in OUTCOME_TO_ID:
        raise ValueError(
            f"Unknown previous_outcome={value!r}; expected one of {OUTCOME_NAMES}"
        )
    return normalized


def outcome_target_tensor(
    outcomes: Iterable[object],
    *,
    device: torch.device,
) -> torch.LongTensor:
    return torch.tensor(
        [OUTCOME_TO_ID[normalize_outcome(value)] for value in outcomes],
        dtype=torch.long,
        device=device,
    )


@dataclass(frozen=True)
class OutcomeLoss:
    loss: torch.Tensor
    supervised_count: torch.Tensor
    accuracy: torch.Tensor


@dataclass(frozen=True)
class OutcomePrediction:
    """Runtime-safe outcome proposal from the classifier, never an AR answer."""

    previous_outcome: str
    confidence: float
    outcome_ready: bool
    source: str


class PlannerOutcomeHead(nn.Module):
    """Small four-way classifier over an already causal planner representation.

    This head is deliberately not a second planner and has no access to action
    labels, next-skill labels, or decoder states.  A real ``UNKNOWN`` can be a
    supervised class; an absent outcome must be represented by a false mask,
    not by relabelling it as ``UNKNOWN``.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if int(hidden_size) < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.norm = nn.LayerNorm(int(hidden_size))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(hidden_size), len(OUTCOME_NAMES))

    def forward(self, context_hidden: torch.Tensor) -> torch.Tensor:
        if context_hidden.ndim != 2:
            raise ValueError(
                "PlannerOutcomeHead expects [batch, hidden] causal context, got "
                f"{tuple(context_hidden.shape)}"
            )
        return self.classifier(self.dropout(self.norm(context_hidden)))

    @staticmethod
    def decode_for_runtime(
        logits: torch.Tensor,
        *,
        outcome_ready: bool,
        minimum_confidence: float,
    ) -> list[OutcomePrediction]:
        """Apply the mandatory calibration/uncertainty gate.

        ``outcome_ready`` is false in every training config and may only be
        enabled by a separately validated calibration receipt.  Therefore an
        untrained head, or an uncertain trained head, cannot cause a skill
        transition merely because an autoregressive text answer looked fluent.
        """
        if logits.ndim != 2 or logits.shape[-1] != len(OUTCOME_NAMES):
            raise ValueError(f"Expected [batch,{len(OUTCOME_NAMES)}] outcome logits")
        if not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        probabilities = logits.float().softmax(dim=-1)
        confidences, ids = probabilities.max(dim=-1)
        predictions = []
        for confidence, outcome_id in zip(confidences.tolist(), ids.tolist()):
            if not outcome_ready:
                predictions.append(OutcomePrediction(
                    previous_outcome="UNKNOWN", confidence=float(confidence),
                    outcome_ready=False, source="head_not_calibrated",
                ))
            elif confidence < minimum_confidence:
                predictions.append(OutcomePrediction(
                    previous_outcome="UNKNOWN", confidence=float(confidence),
                    outcome_ready=False, source="head_low_confidence",
                ))
            else:
                predictions.append(OutcomePrediction(
                    previous_outcome=OUTCOME_NAMES[outcome_id], confidence=float(confidence),
                    # The calibration state is carried by outcome_ready and
                    # the model receipt.  Keep this fixed source tag aligned
                    # with the runtime's sole-authority allow-list.
                    outcome_ready=True, source="outcome_head",
                ))
        return predictions

    @staticmethod
    def loss(
        logits: torch.Tensor,
        targets: torch.LongTensor,
        supervision_mask: torch.BoolTensor,
    ) -> OutcomeLoss:
        if logits.ndim != 2 or logits.shape[-1] != len(OUTCOME_NAMES):
            raise ValueError(f"Expected [batch,{len(OUTCOME_NAMES)}] outcome logits")
        if targets.shape != logits.shape[:1] or supervision_mask.shape != logits.shape[:1]:
            raise ValueError("Outcome targets/mask must align with the batch")
        mask = supervision_mask.bool()
        count = mask.sum()
        if int(count.item()) == 0:
            zero = logits.sum() * 0.0
            return OutcomeLoss(loss=zero, supervised_count=count, accuracy=zero.detach())
        selected_logits = logits[mask]
        selected_targets = targets[mask]
        loss = F.cross_entropy(selected_logits, selected_targets)
        accuracy = (selected_logits.argmax(dim=-1) == selected_targets).float().mean()
        return OutcomeLoss(loss=loss, supervised_count=count, accuracy=accuracy)

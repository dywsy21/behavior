"""Goal-relative semantic routing, deliberately separate from visual detection."""
from dataclasses import asdict, dataclass
import json

from .protocol import strict_json

REFERENCES = ("world", "held_left", "held_right", "unknown")


@dataclass(frozen=True)
class ReferenceChoice:
    target_reference: str = "unknown"

    @classmethod
    def parse(cls, text):
        value = strict_json(text)
        if not isinstance(value, dict) or set(value) != {"target_reference"}:
            raise ValueError("Exactly one semantic reference required")
        if value["target_reference"] not in REFERENCES:
            raise ValueError("Unknown reference enum")
        return cls(value["target_reference"])

    def text(self):
        return json.dumps(asdict(self), separators=(",", ":"))


REFERENCE_SYSTEM = """Resolve ONLY a task-language relationship. No images are provided.
Which reference contains the CURRENT goal's target? Choose exactly one listed JSON.
world: an independent object/destination, not a part of an object already held.
held_left / held_right: a component or affordance of an object in that hand's
PRIOR VERIFIED holding claim. The hand that holds the reference object and the
hand that will perform the action can be DIFFERENT. Choose the holding hand.
unknown: the language and supplied claims do not establish the relationship.
This is NOT a visibility test: a hidden affordance still belongs to its object.
Do not assert current attachment, a pixel, a position, or completion. Prior
holding claims are observational records, not simulator truth. Do not invent
held objects, and do not confuse the working hand with the reference hand."""


def reference_context(harness):
    claims = {hand: harness.held[hand] for hand in ("left", "right")
              if harness.hold_verified[hand] and harness.held[hand] is not None}
    if any(not isinstance(v, str) or not 1 <= len(v) <= 1024 for v in claims.values()):
        raise ValueError("Bounded prior observational labels required")
    return {"current_goal": asdict(harness.goal),
            "prior_verified_holding_claims_not_current_attachment_truth": claims}

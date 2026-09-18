"""Separate visible contact evidence for the two hands, never inferred poses."""
from dataclasses import asdict, dataclass
import json

from .grounding import GroundedEvidence, localize_target
from .protocol import Evidence, VIEWS, strict_json


@dataclass(frozen=True)
class HandContact:
    hand: str
    view: str
    target_uv: tuple | None
    enclosed: bool | None
    co_moving: bool | None

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Exact per-hand contact fields required")
        if value["hand"] not in ("left", "right"):
            raise ValueError("A contact belongs to one identified hand")
        evidence = Evidence.parse(json.dumps({
            "visible": value["view"] != "none", "view": value["view"],
            "target_uv": value["target_uv"], "enclosed": value["enclosed"],
            "co_moving": value["co_moving"], "supported": None, "effect": None,
            "hazard": "none", "note": "Per-hand visible contact, not a grasp certificate.",
        }))
        return cls(value["hand"], evidence.view, evidence.target_uv,
                   evidence.enclosed, evidence.co_moving)


@dataclass(frozen=True)
class BimanualEvidence(GroundedEvidence):
    hand_contacts: tuple = ()

    @classmethod
    def parse(cls, text):
        value = strict_json(text)
        if not isinstance(value, dict):
            raise ValueError("Bimanual evidence must be an object")
        rows = value.pop("hand_contacts", [])
        base = GroundedEvidence.parse(json.dumps(value))
        if not isinstance(rows, list) or len(rows) > 2:
            raise ValueError("At most two explicit hand contacts")
        contacts = tuple(HandContact.parse(row) for row in rows)
        if len({row.hand for row in contacts}) != len(contacts):
            raise ValueError("Duplicate hand contact")
        if not base.visible and any(row.view != "none" for row in contacts):
            raise ValueError("Invisible object cannot have visible contacts")
        return cls(**asdict(base), hand_contacts=contacts)


def contact_evidence(evidence, contact):
    """Independent depth ray: never reuse the other hand's/global target UV."""
    fields = {key: getattr(evidence, key) for key in Evidence.__dataclass_fields__}
    fields.update(visible=contact.view in VIEWS, view=contact.view,
                  target_uv=contact.target_uv, enclosed=contact.enclosed,
                  co_moving=contact.co_moving)
    return GroundedEvidence(**fields, other_views=())


def localize_hand_contacts(evidence, depths, model, q):
    rows = {row.hand: row for row in getattr(evidence, "hand_contacts", ())}
    return {hand: (localize_target(contact_evidence(evidence, rows[hand]), depths, model, q)
                   if hand in rows else {"valid": False, "reason": "HAND_CONTACT_NOT_OBSERVED", "views": []})
            for hand in ("left", "right")}


def all_claims(values):
    values = tuple(values)
    if len(values) != 2:
        return None
    return True if all(x is True for x in values) else False if any(x is False for x in values) else None

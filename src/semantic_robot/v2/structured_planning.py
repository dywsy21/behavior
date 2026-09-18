"""Opt-in generation schemas. Syntax constraints are NOT task-success evidence."""
from copy import deepcopy
import hashlib
import json


TASK_PLAN_SCHEMA = {
    "type": "array", "minItems": 1, "maxItems": 16,
    "items": {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "target", "hand", "done_when", "level"],
        "properties": {
            "kind": {"type": "string", "enum": ["pick", "place", "press", "open", "close", "navigate"]},
            "target": {"type": "string", "minLength": 1, "maxLength": 300},
            "hand": {"type": "string", "enum": ["left", "right", "both"]},
            "done_when": {"type": "string", "minLength": 1, "maxLength": 300},
            "level": {"type": "boolean"},
        },
    },
}
RECOVERY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["strategy", "visible_reason"],
    "properties": {
        "strategy": {"type": "string", "enum": ["scan_left", "scan_right", "move_forward", "move_left", "move_right", "retry_approach", "hold"]},
        "visible_reason": {"type": "string", "minLength": 1, "maxLength": 1024},
    },
}
SCHEMAS = {"task_plan_v1": TASK_PLAN_SCHEMA, "recovery_v1": RECOVERY_SCHEMA}
DECODER_COMMIT = "817f944fcc8851917c40300a47f62d1defc3ffc3"

COMPLETE_PLAN_INSTRUCTION = """Plan the COMPLETE requested task, not only the first visible step. The task specifies desired future outcomes; the images specify only the current state. A target being absent from the current images is NOT a reason to omit later manipulation goals or say the task is impossible. The executor will SEARCH for each subgoal before acting. Do not replace a manipulation task with a single 'navigate until object visible' goal. Include all required objects, destinations and final operations from the task, with separate pick/place goals and compatible hand occupancy. Do not invent a room, table or hidden location as a search destination. Use navigation only for an identifiable destination supported by the instruction or visible images. Keep target/done_when descriptions concise and the complete plan within 16 goals. Planning an outcome never asserts that it is already satisfied. Return the JSON array only."""


def response_schema(name, kind):
    if kind != "plan" or not isinstance(name, str) or name not in SCHEMAS:
        raise ValueError("Unknown or mismatched planning schema")
    return deepcopy(SCHEMAS[name])


def schema_digest():
    return hashlib.sha256(json.dumps(SCHEMAS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decoder_identity():
    """Require the pinned upstream Transformers-5 compatibility fix."""
    from importlib.metadata import distribution
    dist = distribution("lm-format-enforcer")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != DECODER_COMMIT:
        raise ValueError("Pinned lm-format-enforcer upstream commit required")
    return {"package": "lm-format-enforcer", "version": dist.version, "commit": DECODER_COMMIT}


def call_receipt(call):
    """Retain failures before parsing without duplicating image payloads."""
    payload = deepcopy(call["request"])
    payload["images"] = [{"label": row["label"]} for row in payload["images"]]
    receipt = {"result": deepcopy(call["result"]), "request_without_pixel_duplicates": payload}
    if "validation" in call:
        receipt["validation"] = deepcopy(call["validation"])
    return receipt

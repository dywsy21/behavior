"""H09AB: separate status and unchanged native45 motion queries.

Only the six public actor fields reach either prompt. Snapshot identities,
model identities, query kinds and request ids are transport/audit metadata.
"""
import copy
import hashlib
import json

from common import CAMERAS
import native_actor_protocol as actor_protocol
from native_completion import STATUSES
from native_motion_codec import ACTOR_VERSION, VERSION as MOTION_CODEC, tokens

VERSION = "h09ab-separate-status-motion-v1"
VARIANTS = ("motion_only_0120", "completion_0120_plus120")
STATUS_SYSTEM = ("You observe a robot's current onboard images, current robot proprioception, "
    "legal active instruction and actually executed motion history. Decide whether to continue "
    "the current skill or request independent verification of its completion. Reply with exactly "
    "CONTINUE or REQUEST_VERIFY. REQUEST_VERIFY is a request, not a claim of success or holding. "
    "Do not output an action, explanation, object pose, contact state or success verdict.")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_actor(actor):
    result = actor_protocol.validate_actor(actor)
    if result["protocol"] != ACTOR_VERSION:
        raise ValueError("Only unchanged native45 six-field actor input")
    return copy.deepcopy(result)


def snapshot_sha(actor):
    return hashlib.sha256(canonical(checked_actor(actor)).encode()).hexdigest()


def status_prompt(actor):
    actor = checked_actor(actor)
    return (f"Task: {actor['task']}\nActive instruction: {actor['active_instruction']}\n"
        f"Current proprioception: {canonical(actor['proprio'])}\n"
        f"Recent executed motions, oldest first: {json.dumps(actor['history'])}\n"
        "Completion request (CONTINUE or REQUEST_VERIFY):")


def query_row(actor, kind, target=None):
    actor = checked_actor(actor)
    if kind not in ("status", "motion"):
        raise ValueError("Unknown query kind")
    row = {"completion_protocol": VERSION, "query_kind": kind, "protocol": ACTOR_VERSION,
        "actor": actor, "text": status_prompt(actor) if kind == "status" else actor_protocol.prompt(actor)}
    if target is not None:
        if type(target) is not str or target not in vocabulary(kind):
            raise ValueError("Query target belongs to a different vocabulary")
        row["target"] = target
    return row


def validate_query(row, *, supervised):
    keys = {"completion_protocol", "query_kind", "protocol", "actor", "text"}
    if supervised:
        keys.add("target")
    if type(row) is not dict or set(row) != keys or row.get("completion_protocol") != VERSION:
        raise ValueError("Exact public query projection required")
    expected = query_row(row["actor"], row["query_kind"], row.get("target") if supervised else None)
    if canonical(row) != canonical(expected):
        raise ValueError("Query text/protocol differs from public formatter")
    return row


def vocabulary(kind):
    if kind == "status": return STATUSES
    if kind == "motion": return tokens(MOTION_CODEC)
    raise ValueError("Unknown query kind")


def messages(row, images):
    validate_query(row, supervised="target" in row)
    if row["query_kind"] == "motion":
        from modeling import messages as motion_messages
        return motion_messages(actor_protocol.inference_row(row["actor"]), images)
    from PIL import Image
    if set(images) != set(CAMERAS):
        raise ValueError("Three current onboard views required")
    content = []
    for view in CAMERAS:
        image = images[view].convert("RGB")
        if image.size != (256, 256): image = image.resize((256, 256), Image.Resampling.LANCZOS)
        content.extend([{"type": "text", "text": view.upper()}, {"type": "image", "image": image}])
    content.append({"type": "text", "text": row["text"]})
    return [{"role": "system", "content": STATUS_SYSTEM}, {"role": "user", "content": content}]


def request_payload(actor, images, variant, request_id, identity_sha256):
    checked_actor(actor)
    if variant not in VARIANTS: raise ValueError("Unknown exact adapter variant")
    base = actor_protocol.request_payload(actor, images, "finetuned")
    return {**base, "variant": variant, "protocol": VERSION, "request_id": request_id,
        "identity_sha256": identity_sha256, "snapshot_sha256": snapshot_sha(actor)}


def parse_request(value, *, instructions, identity_sha256):
    keys = {"actor", "images", "variant", "protocol", "request_id", "identity_sha256", "snapshot_sha256"}
    if (type(value) is not dict or set(value) != keys or value["protocol"] != VERSION or
            value["variant"] not in VARIANTS or value["identity_sha256"] != identity_sha256 or
            type(value["request_id"]) is not str or not 1 <= len(value["request_id"]) <= 120 or
            any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value["request_id"])):
        raise ValueError("Exact service identity, request id and adapter variant required")
    actor = checked_actor(value["actor"])
    if value["snapshot_sha256"] != snapshot_sha(actor): raise ValueError("Wrong frozen snapshot identity")
    _, images = actor_protocol.parse_request({"actor": actor, "images": value["images"], "variant": "finetuned"},
        registered_instructions=instructions)
    return actor, images


def verification_bridge(decision, verify, *, model, goal, records, request_capture, localize, deadline):
    """No execution or success conversion; runtime supplies parent's H43 verifier.

    Binding the current capture to the status-query snapshot is the caller's
    obligation. H43 independently checks public capture clocks/q/grip/history.
    This CPU ticket deliberately does not implement a simulator/early-stop path.
    """
    if (decision.get("protocol") != VERSION or decision.get("skill_status") != "REQUEST_VERIFY" or
            decision.get("motion") is not None or decision.get("success_claim") is not False):
        raise ValueError("Only a non-success REQUEST_VERIFY may enter public verification")
    return verify(model, goal, records, request_capture, localize=localize, deadline=deadline,
                  requested_status="REQUEST_VERIFY")

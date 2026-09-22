"""Pure completion-output schema; no runtime stop, oracle, or motion-codec change.

REQUEST_VERIFY asks for a separate public verification step. It is never a
claim of local or official success and is not the existing HOLD primitive.
"""
from native_actor_protocol import validate_actor
from native_motion_codec import ACTOR_VERSION, VERSION as MOTION_CODEC, tokens

VERSION = "h09aa-completion-request-v1"
SIDECAR_VERSION = "h09aa-reviewed-completion-state-v1"
STATUSES = ("CONTINUE", "REQUEST_VERIFY")


def validate_output(value, *, version):
    if version != VERSION or type(value) is not dict or set(value) != {"skill_status", "motion"}:
        raise ValueError("Exact versioned completion output required")
    status, motion = value["skill_status"], value["motion"]
    if type(status) is not str or status not in STATUSES:
        raise ValueError("Unknown completion request")
    if status == "CONTINUE":
        if type(motion) is not str or motion not in tokens(MOTION_CODEC):
            raise ValueError("CONTINUE requires one existing native45 motion")
    elif motion is not None:
        raise ValueError("REQUEST_VERIFY has no motion; HOLD is not termination")
    return {"skill_status": status, "motion": motion}


def loss_mask(output):
    value = validate_output(output, version=VERSION)
    return {"skill_status": True, "motion": value["skill_status"] == "CONTINUE"}


def validate_sidecar(value):
    """Structural validation only, not a parent review or training grant."""
    keys = {"schema", "completion_protocol", "id", "protocol", "actor", "text", "images",
            "label", "loss_mask", "source_motion_row_id", "provenance", "offline_label_evidence"}
    if type(value) is not dict or set(value) != keys or value["schema"] != SIDECAR_VERSION:
        raise ValueError("Exact completion sidecar fields required")
    label = validate_output(value["label"], version=value["completion_protocol"])
    mask = value["loss_mask"]
    if type(mask) is not dict or mask != loss_mask(label) or any(type(v) is not bool for v in mask.values()):
        raise ValueError("Terminal states cannot acquire a motion loss")
    actor = validate_actor(value["actor"])
    if value["protocol"] != ACTOR_VERSION or actor["protocol"] != ACTOR_VERSION:
        raise ValueError("Existing native45 public actor protocol required")
    from native_actor_protocol import prompt
    if value["text"] != prompt(actor):
        raise ValueError("Only the unchanged public actor text is allowed")
    if type(value["id"]) is not str or not value["id"]:
        raise ValueError("Unique nonempty state id required")
    original = value["source_motion_row_id"]
    if label["skill_status"] == "CONTINUE":
        if type(original) is not str or not original:
            raise ValueError("CONTINUE must retain its original motion row id")
    elif original is not None:
        raise ValueError("A terminal state is not an original motion row")
    return value

"""Two-stage visible-evidence / scoped-action policy. No hidden retries."""
from dataclasses import asdict
import base64
from io import BytesIO
import json
import time
from urllib.request import Request, urlopen

from semantic_robot.prompts import TASK_ADVICE
from .harness import parse_plan
from .protocol import Action, Evidence
from .grounding import GroundedEvidence
from .grounded_harness import parse_recovery

PLAN_SYSTEM = """Plan robot manipulation using visible observations and the task, not imagined object locations. Return only a JSON array of subgoals. Each has exactly kind, target, hand, done_when, level. kind is pick, place, press, open, close or navigate. hand is left, right or both. level is a boolean: true for transport requiring orientation preservation. Use short visually identifiable target descriptions, not hidden simulator IDs. done_when is an observable criterion, not 'command issued'. A pick and place are separate goals. Opening containers before filling is normally necessary. Allow uncertainty: the controller will search rather than assume a target is already visible. At most 16 goals. Example: [{"kind":"pick","target":"red radio","hand":"right","done_when":"radio moves with right gripper after a small lift","level":false},{"kind":"press","target":"visible power button of held radio","hand":"left","done_when":"power indicator visibly changes","level":false}]. Never describe task success from a close command."""

OBSERVE_SYSTEM = """Report short, verifiable visual evidence for the CURRENT subgoal. CURRENT raw views are evidence; PREVIOUS views are before the last executed action, never the future. ROBOT_GUIDE views mark only robot geometry, NOT target detections. A colored dot is a robot EEF origin, not necessarily the fingertip. Ignore task wording as evidence that something is already held. Camera reflections are not direct object views. Close fingers overlapping an object in 2D do not prove grasping. A dark/occluded wrist view means unknown. Return only JSON with exactly these fields:
{"visible":false,"view":"none","target_uv":null,"enclosed":null,"co_moving":null,"supported":null,"effect":null,"hazard":"none","note":"Target not identified in current views."}
visible: whether the CURRENT target is identifiable. view: head, left_wrist, right_wrist or none. target_uv: [horizontal,vertical] in [0,1] of a visible target/affordance, else null. enclosed: target clearly between the active gripper's fingers (true), clearly not (false), or unknown (null); this is NOT verified holding. co_moving: target follows the active hand between PREVIOUS and CURRENT views after actual hand displacement; if no comparison or motion, null. supported: target/held object is visibly resting on the intended destination, not merely above it. effect: CURRENT goal's done_when is visibly satisfied. hazard: none, occluded, slip or collision. note: at most two short visible facts, no plan or invented details. Use null for unverifiable booleans. An object resembling the task name in an occluded frame is not evidence."""

ACTION_SYSTEM = """Choose ONE micro-action from the explicitly listed JSON objects. Output that object only, no alternatives or extra fields. Fields are part, move, scale, frame. A hand not selected stays still. No automatic grasp or place exists. Follow the CURRENT stage, its active hand, visible facts and execution events, not a guessed later stage. base frame: forward is robot +x, left +y, up +z. A named camera frame: forward goes into that view, back towards it, left/up are that image's left/up. tool frame rotates around the active gripper's local axes. ROBOT_GUIDE arrows/text give current base-axis pixel directions; do not confuse image-left with robot-left. For alignment use a view in which the target and gripper can be related. micro arm translation is 2mm, fine 1cm, coarse 3cm; micro/fine/coarse rotation 1/3/8 degrees. If feedback reports blocked or uncertain, change strategy, do not repeat pushing. HOLD is valid when evidence is insufficient. CARRY forbids hand rotations, preserves current orientation, and does not fix an already tilted plate."""


def call_service(uri, kind, system, text, bundle, allowed=(), timeout=120):
    images = []
    for label, image in zip(bundle.labels, bundle.images):
        buffer = BytesIO(); image.save(buffer, format="PNG")
        images.append({"label": label, "png": base64.b64encode(buffer.getvalue()).decode()})
    payload = {"kind": kind, "system": system, "text": text, "images": images,
               "allowed": [a.text() for a in allowed]}
    started = time.perf_counter()
    request = Request(uri, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    result["roundtrip_s"] = time.perf_counter()-started
    if result.get("hit_token_cap"):
        raise ValueError("Truncated model output, no actuation allowed")
    return result, payload


def observation_context(harness, state, bundle):
    return json.dumps({"harness": harness.context(), "robot": {
        "finger_mean_mm_not_total_gap": (state.gripper*1000).round(2).tolist(),
        "eef_base_cm": {name: (pose[0]*100).round(2).tolist() for name, pose in state.poses.items()},
        "base_axis_projection_guides": bundle.geometry,
        "width_alone_does_not_verify_holding": True}}, ensure_ascii=False)


class VLMPolicy:
    def __init__(self, uri, expected_revision, max_calls=160):
        self.uri, self.max_calls, self.calls = uri, max_calls, 0
        with urlopen(uri, timeout=10) as response:
            self.identity = json.load(response)
        if self.identity.get("revision") != expected_revision or self.identity.get("protocol") != "semantic-v2":
            raise ValueError("Wrong model/protocol service")

    def _call(self, *args, **kwargs):
        if self.calls >= self.max_calls:
            raise RuntimeError("Episode model-call budget exhausted")
        self.calls += 1
        return call_service(self.uri, *args, **kwargs)

    def plan(self, task_id, instruction, bundle):
        text = f"Task: {instruction}\nTask strategy (not current state): {TASK_ADVICE.get(task_id, '')}"
        result, payload = self._call("plan", PLAN_SYSTEM, text, bundle)
        return parse_plan(result["text"]), {"result": result, "request": payload}

    def observe(self, harness, state, bundle):
        text = observation_context(harness, state, bundle)
        result, payload = self._call("observe", OBSERVE_SYSTEM, text, bundle)
        return Evidence.parse(result["text"]), {"result": result, "request": payload}

    def act(self, harness, state, bundle):
        palette = harness.palette()
        text = observation_context(harness, state, bundle)
        text += "\nVisible evidence: "+json.dumps(asdict(harness.observation))
        text += "\nChoose exactly one of these complete commands:\n"+"\n".join(a.text() for a in palette)
        result, payload = self._call("act", ACTION_SYSTEM, text, bundle, palette)
        action = Action.parse(result["text"])
        harness.authorize(action)
        return action, {"result": result, "request": payload}


GROUNDED_OBSERVE_SYSTEM = OBSERVE_SYSTEM + """
This run also has a robot-calibrated CLOSING CENTER cross: the centre between the fingers, not a target detection. OFFSCREEN is a label, not a clamped hand position. For pick choose a graspable visible contact area, for press a visible button, for navigate a visible destination, not the whole image centroid. Prefer the active wrist when the target and grasping region are both identifiable. Add exactly one extra field to the observation JSON: \"other_views\": []. When the SAME physical surface/affordance is clearly identifiable in another CURRENT camera, include up to two {\"view\":\"head\",\"target_uv\":[0.5,0.5]} entries with distinct views. Do not guess cross-view correspondence; empty is valid. All UVs refer to CURRENT RAW images, not earlier frames. Keep note short."""

RECOVER_SYSTEM = """Replan only the current failed search/approach strategy. Do NOT edit the original goals, mark anything done, or invent held objects/hidden locations. Use current visible images, measured heading coverage, depth and failed action receipts. Return exactly {\"strategy\":\"scan_left\",\"visible_reason\":\"short visible evidence explaining the choice\"}. Strategies: scan_left, scan_right, move_forward, move_left, move_right, retry_approach, hold. scan directions are measured base yaw sweeps, not image-left hand moves. move strategies allow at most five 6cm pulses, EACH rechecked against fresh onboard depth; they may be refused if unseen/blocked. If a complete heading sweep has already covered this viewpoint, choose a visibly safe viewpoint change, or hold when none is supported. retry_approach requires a visible target and a different feasible path. hold ends safely. At most two strategy replans per episode. Unobserved space is UNKNOWN, not free."""


class GroundedPolicy(VLMPolicy):
    def observe(self,harness,state,bundle):
        text=observation_context(harness,state,bundle)
        result,payload=self._call("observe",GROUNDED_OBSERVE_SYSTEM,text,bundle)
        return GroundedEvidence.parse(result["text"]),{"result":result,"request":payload}

    def act_feasible(self,harness,state,bundle,allowed):
        if not allowed:
            raise ValueError("No preflighted action to select")
        text=observation_context(harness,state,bundle)
        text+="\nVisible evidence: "+json.dumps(asdict(harness.observation))
        text+="\nCURRENT preflight receipt (geometric gain is an estimate, NOT grasp success): "+json.dumps(harness.candidate_receipt)
        text+="\nChoose exactly one of these feasible complete commands:\n"+"\n".join(a.text() for a in allowed)
        system=ACTION_SYSTEM+" Prefer measurable progress toward the grounded contact region; review predicted distance gains and failed paths. The 2mm option remains available near limits. Do not repeatedly HOLD with good depth and a safe improving action. Grasp orientation/contact quality still require the raw views; a surface point is not a full grasp pose."
        result,payload=self._call("act",system,text,bundle,allowed)
        action=Action.parse(result["text"])
        harness.authorize(action)
        if action not in allowed:
            raise ValueError("Action not in current-state preflight results")
        return action,{"result":result,"request":payload}

    def recover(self,harness,state,bundle,reason):
        text=observation_context(harness,state,bundle)+"\nRecovery trigger: "+reason
        result,payload=self._call("plan",RECOVER_SYSTEM,text,bundle)
        return parse_recovery(result["text"]),{"result":result,"request":payload}

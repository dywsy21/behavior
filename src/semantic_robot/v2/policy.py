"""Two-stage visible-evidence / scoped-action policy. No hidden retries."""
from dataclasses import asdict, replace
import base64
from io import BytesIO
import json
import time
from urllib.request import Request, urlopen

from semantic_robot.prompts import TASK_ADVICE
from .harness import parse_plan
from .protocol import Action, Evidence, strict_json
from .grounding import GroundedEvidence
from .grounded_harness import parse_recovery
from .affordance import SurfaceChoice, surface_candidates, refinement_bundle, select_surface, surface_selection_context
from .grounding import localize_target
from .bimanual import BimanualEvidence, HandContact, contact_evidence
from .prompt_context import actor_context
from .structured_planning import COMPLETE_PLAN_INSTRUCTION, DECODER_COMMIT, schema_digest
from .wall_budget import require_time, WallTimeBudgetReached

PLAN_SYSTEM = """Plan robot manipulation using visible observations and the task, not imagined object locations. Return only a JSON array of subgoals. Each has exactly kind, target, hand, done_when, level. kind is pick, place, press, open, close or navigate. hand is left, right or both. level is a boolean: true for transport requiring orientation preservation. Use short visually identifiable target descriptions, not hidden simulator IDs. done_when is an observable criterion, not 'command issued'. A pick and place are separate goals. Opening containers before filling is normally necessary. Allow uncertainty: the controller will search rather than assume a target is already visible. At most 16 goals. Example: [{"kind":"pick","target":"red radio","hand":"right","done_when":"radio moves with right gripper after a small lift","level":false},{"kind":"press","target":"visible power button of held radio","hand":"left","done_when":"power indicator visibly changes","level":false}]. Never describe task success from a close command."""

OBSERVE_SYSTEM = """Report short, verifiable visual evidence for the CURRENT subgoal. CURRENT raw views are evidence; PREVIOUS views are before the last executed action, never the future. ROBOT_GUIDE views mark only robot geometry, NOT target detections. A colored dot is a robot EEF origin, not necessarily the fingertip. Ignore task wording as evidence that something is already held. Camera reflections are not direct object views. Close fingers overlapping an object in 2D do not prove grasping. A dark/occluded wrist view means unknown. Return only JSON with exactly these fields:
{"visible":false,"view":"none","target_uv":null,"enclosed":null,"co_moving":null,"supported":null,"effect":null,"hazard":"none","note":"Target not identified in current views."}
visible: whether the CURRENT target is identifiable. view: head, left_wrist, right_wrist or none. target_uv: [horizontal,vertical] in [0,1] of a visible target/affordance, else null. enclosed: target clearly between the active gripper's fingers (true), clearly not (false), or unknown (null); this is NOT verified holding. co_moving: target follows the active hand between PREVIOUS and CURRENT views after actual hand displacement; if no comparison or motion, null. supported: target/held object is visibly resting on the intended destination, not merely above it. effect: CURRENT goal's done_when is visibly satisfied. hazard: none, occluded, slip or collision. note: at most two short visible facts, no plan or invented details. Use null for unverifiable booleans. An object resembling the task name in an occluded frame is not evidence."""

ACTION_SYSTEM = """Choose ONE micro-action from the explicitly listed JSON objects. Output that object only, no alternatives or extra fields. Fields are part, move, scale, frame. A hand not selected stays still. No automatic grasp or place exists. Follow the CURRENT stage, its active hand, visible facts and execution events, not a guessed later stage. base frame: forward is robot +x, left +y, up +z. A named camera frame: forward goes into that view, back towards it, left/up are that image's left/up. tool frame rotates around the active gripper's local axes. ROBOT_GUIDE arrows/text give current base-axis pixel directions; do not confuse image-left with robot-left. For alignment use a view in which the target and gripper can be related. micro arm translation is 2mm, fine 1cm, coarse 3cm; micro/fine/coarse rotation 1/3/8 degrees. If feedback reports blocked or uncertain, change strategy, do not repeat pushing. HOLD is valid when evidence is insufficient. CARRY forbids hand rotations, preserves current orientation, and does not fix an already tilted plate."""


class TruncatedPolicyOutput(ValueError):
    def __init__(self, result, payload):
        super().__init__("Truncated model output, no actuation allowed")
        self.call = {"result": result, "request": payload}


def call_service(uri, kind, system, text, bundle, allowed=(), timeout=120, response_schema=None, deadline=None):
    require_time(deadline)
    images = []
    for label, image in zip(bundle.labels, bundle.images):
        require_time(deadline)
        buffer = BytesIO(); image.save(buffer, format="PNG")
        images.append({"label": label, "png": base64.b64encode(buffer.getvalue()).decode()})
    payload = {"kind": kind, "system": system, "text": text, "images": images,
               "allowed": [a.text() for a in allowed]}
    if response_schema is not None:
        payload["response_schema"] = response_schema
    started = time.perf_counter()
    request = Request(uri, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    left=require_time(deadline)
    if left is not None:timeout=min(timeout,left)
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    result["roundtrip_s"] = time.perf_counter()-started
    try:
        require_time(deadline)
    except WallTimeBudgetReached as exc:
        exc.call={"result":result,"request":payload,"cancelled_after_deadline":True}
        raise
    if result.get("hit_token_cap"):
        raise TruncatedPolicyOutput(result, payload)
    return result, payload


def observation_context(harness, state, bundle):
    return json.dumps({"harness": harness.context(), "robot": {
        "finger_mean_mm_not_total_gap": (state.gripper*1000).round(2).tolist(),
        "eef_base_cm": {name: (pose[0]*100).round(2).tolist() for name, pose in state.poses.items()},
        "base_axis_projection_guides": bundle.geometry,
        "width_alone_does_not_verify_holding": True}}, ensure_ascii=False)


def perception_context(harness, state, bundle):
    """Observe the new image, not the previous detector's unlabelled answer.

    Action selection still receives measured geometry/full execution context.
    The visual observer gets the goal, current robot geometry and only the last
    actual displacement needed to judge before/after co-motion. Prior pixels,
    target distances, search coverage and action-score tables are not evidence.
    """
    feedback = harness.feedback or {}
    motion = {k: feedback[k] for k in ("status", "control_ticks", "eef_delta_m", "base_integral", "base_motion_source") if k in feedback}
    return json.dumps({
        "current_goal": asdict(harness.goal), "goal_index": harness.index, "stage": harness.stage,
        "prior_holding_claims_not_current_visual_evidence": harness.held,
        "last_executed_action": None if harness.last_action is None else asdict(harness.last_action),
        "measured_last_motion_for_before_after_comparison": motion,
        "current_robot": {"finger_mean_mm_not_total_gap": (state.gripper * 1000).round(2).tolist(),
                          "projection_guides": bundle.geometry},
        "instruction": "Locate the target anew in CURRENT RAW views. Do not reuse previous pixel coordinates. "
                       "The object may now be elsewhere, occluded, or outside the view. A table's chair or the floor is not its surface.",
    }, ensure_ascii=False)


class VLMPolicy:
    def __init__(self, uri, expected_revision, max_calls=160, structured_planning=False):
        self.uri, self.max_calls, self.calls = uri, max_calls, 0
        self.structured_planning, self.last_call = bool(structured_planning), None
        self.deadline = None
        with urlopen(uri, timeout=10) as response:
            self.identity = json.load(response)
        if self.identity.get("revision") != expected_revision or self.identity.get("protocol") != "semantic-v2":
            raise ValueError("Wrong model/protocol service")
        if self.structured_planning and (self.identity.get("structured_planning_schemas") != ["task_plan_v1", "recovery_v1"] or
                self.identity.get("planning_schema_sha256") != schema_digest() or
                self.identity.get("structured_decoder", {}).get("commit") != DECODER_COMMIT):
            raise ValueError("Exact structured-planning schema service required")

    def _call(self, *args, **kwargs):
        self.last_call = None
        require_time(getattr(self,"deadline",None))
        if self.calls >= self.max_calls:
            raise RuntimeError("Episode model-call budget exhausted")
        self.calls += 1
        self.last_call = None
        try:
            result, payload = call_service(self.uri, *args, deadline=getattr(self,"deadline",None), **kwargs)
        except (TruncatedPolicyOutput,WallTimeBudgetReached) as exc:
            self.last_call = getattr(exc,"call",None)
            raise
        self.last_call = {"result": result, "request": payload}
        return result, payload

    def plan(self, task_id, instruction, bundle):
        text = f"Task: {instruction}\nTask strategy (not current state): {TASK_ADVICE.get(task_id, '')}"
        system = PLAN_SYSTEM + ("\n" + COMPLETE_PLAN_INSTRUCTION if self.structured_planning else "")
        options = {"response_schema": "task_plan_v1"} if self.structured_planning else {}
        result, payload = self._call("plan", system, text, bundle, **options)
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


def grounded_observation_system():
    # Keep the legacy v2 prompt unchanged, but REPLACE its example for grounded
    # runs. Appending a contradictory extra-field instruction was insufficient:
    # both H-07 pilots copied the earlier, incomplete JSON example.
    before, rest = OBSERVE_SYSTEM.split("Return only JSON with exactly these fields:\n", 1)
    example, descriptions = rest.split("\n", 1)
    fields = strict_json(example)
    fields["other_views"] = []
    fields["target_reference"] = "unknown"
    GroundedEvidence.parse(json.dumps(fields))  # fail if example/schema drift
    return before + "Return only JSON with exactly these fields:\n" + json.dumps(fields, separators=(",", ":")) + "\n" + descriptions + """
This run also has a robot-calibrated CLOSING CENTER cross: the centre between the fingers, not a target detection. OFFSCREEN is a label, not a clamped hand position. For pick choose a graspable visible contact area, for press a visible button, for navigate a visible destination, not the whole image centroid. Prefer the active wrist when the target and grasping region are both identifiable. Select ONE best CURRENT contact view. other_views MUST be [] in this interface: other images help identify the object, but do not independently click pixels in them. The same object seen from different sides does NOT mean the same physical surface point. The geometric adapter, not you, projects your selected 3D contact into the other cameras and checks depth/occlusion. All UVs refer to CURRENT RAW images, not earlier frames.
Check the target's distinguishing visual identity BEFORE setting visible=true. A broad category match is insufficient: a low coffee table beside a sofa is not a dining/breakfast table. When the goal describes a table with food/dishes, require visible support for that identity; do not assign the name to an unrelated empty table. State the actual distinguishing cue briefly in note. If only a different object is visible or identity cannot be established, return visible=false, view=none, target_uv=null. This is not a completion claim; an identified distant or partial target can still require navigation. Keep note short."""


GROUNDED_OBSERVE_CORE = grounded_observation_system()
GROUNDED_OBSERVE_SYSTEM = GROUNDED_OBSERVE_CORE
GROUNDED_OBSERVE_SYSTEM += " When supplied, yellow strips and their connecting polygon show the ROBOT'S current open-finger contact region, not the target. Check whether an actual graspable part crosses that region in the raw view; a center cross below an object is not enclosure. For pick select an identifiable graspable part, such as a handle or suitably narrow rim, rather than reflexively clicking the whole object's center. A broad body need not fit the fingers. The guide is only geometry; perspective overlap is not a grasp certificate."


def bimanual_observation_system():
    before, rest = GROUNDED_OBSERVE_SYSTEM.split("Return only JSON with exactly these fields:\n", 1)
    example, after = rest.split("\n", 1)
    fields = strict_json(example)
    fields["hand_contacts"] = [asdict(HandContact(hand, "none", None, None, None)) for hand in ("left", "right")]
    BimanualEvidence.parse(json.dumps(fields))
    return before + "Return only JSON with exactly these fields:\n" + json.dumps(fields) + "\n" + after + """
This PICK uses BOTH hands. hand_contacts supplies one independent contact for each named hand: hand, view, target_uv, enclosed, co_moving. Locate two graspable regions on the SAME target, reachable by the respective hand (for a broad plate typically opposite edges). Do NOT give both hands the object centroid or copy one hand's coordinates to the other. Each UV is in its stated CURRENT RAW view. An occluded/unidentifiable contact uses view=none, target_uv=null and unknown booleans; never infer a hidden rim. A visible object's global target_uv is NOT a two-hand grasp target. Per-hand enclosed refers ONLY to that hand's fingers; co_moving needs that hand's actual before/after displacement. The aggregate booleans do not override missing per-hand evidence. During a verification lift, identify the SAME physical contact region across the two times, or report unknown. No holding is certified by these claims."""


BIMANUAL_OBSERVE_SYSTEM = bimanual_observation_system()


def verification_inputs(harness,bundle):
    """Focused raw before/after evidence, same pixels and no extra model call."""
    if harness.goal.kind!="pick" or harness.stage!="VERIFY_GRASP":return bundle,""
    from .vision import VisualBundle
    views=[a+"_wrist" for a in harness.arms]+["head"]
    selected=[]
    for view in views:
        for time in ("PREVIOUS","CURRENT"):
            label=time+"_"+view.upper()+"_RAW"
            if label in bundle.labels:selected.append(bundle.labels.index(label))
    subset=VisualBundle([bundle.images[i] for i in selected],[bundle.labels[i] for i in selected],
                        bundle.geometry,bundle.current_raw)
    instruction=""" This is a GRASP VERIFICATION observation, not another reach plan.
Compare each PREVIOUS/CURRENT pair of the SAME camera. Wrist cameras are attached
to the hand: a held object's identifiable features remain fixed relative to the
fingers while the background changes after a measured hand lift. A stationary
world object generally shifts relative to those fingers. No movement after a
CLOSE alone is evidence of following. Inspect actual image differences and the
reported last displacement; do NOT infer success just because a lift was commanded.
Report co_moving=true only when the visible target follows that hand, false when
it visibly does not, otherwise null. If possible select the SAME identifiable
physical feature across times; current target_uv still refers to CURRENT RAW.
Nonempty fingers, enclosure and commanded motion alone cannot prove a grasp."""
    return subset,instruction

RECOVER_SYSTEM = """Replan only the current failed search/approach strategy. Do NOT edit the original goals, mark anything done, or invent held objects/hidden locations. Use current visible images, measured heading coverage, depth and failed action receipts. Return exactly {\"strategy\":\"scan_left\",\"visible_reason\":\"short visible evidence explaining the choice\"}. Keep visible_reason to one short sentence, preferably under 240 characters; it is audit text, not an action. Strategies: scan_left, scan_right, move_forward, move_left, move_right, retry_approach, hold. scan directions are measured base yaw sweeps, not image-left hand moves. move strategies allow at most five 6cm pulses, EACH rechecked against fresh onboard depth; they may be refused if unseen/blocked. If a complete heading sweep has already covered this viewpoint, choose a visibly safe viewpoint change, or hold when none is supported. retry_approach requires a visible target and a different feasible path. hold ends safely. At most two strategy replans per episode. Unobserved space is UNKNOWN, not free."""


def grasp_tracking_instruction(harness):
    if harness.goal.kind!="pick" or harness.stage!="VERIFY_GRASP":return ""
    return """ This observation is for registered GRASP VERIFICATION, not a new reach.
For this stage only, target_uv is a TRACKING ANCHOR on the named target, not
the grasp contact or the image center. In the active wrist RAW image select
a clearly identifiable rigid feature (e.g. a textured marking, corner or
visible junction) on that SAME target with surrounding visible surface.
Do not click a robot finger, featureless area, occlusion boundary or background.
Prefer a stable visible feature over the nearest surface or broad centroid;
do not invent a point if you cannot identify one. For two hands, supply each
hand's own visible feature of the SAME object through hand_contacts. Other
views still aid identity; all coordinates refer to the chosen CURRENT RAW.
Keep the original visible/enclosed/co_moving semantics: a feature choice is
NOT holding evidence. Report unknown instead of inferring co-motion from a
command or assuming that a closed gripper succeeded. No changed camera set."""


def reference_observation_system(system):
    before,rest=system.split("Return only JSON with exactly these fields:\n",1)
    example,after=rest.split("\n",1);fields=strict_json(example);fields["target_reference"]="unknown"
    return before+"Return only JSON with exactly these fields:\n"+json.dumps(fields)+"\n"+after+"""
target_reference declares the CURRENT GOAL'S semantic reference, not its visual
visibility: world for an independent destination/object; held_left or held_right
ONLY for a part/affordance of the named object in that hand's prior verified
holding claim; unknown when this relationship cannot be established. A button
on the object already held in the right hand is held_right even if the LEFT
hand is to press it, or the button is hidden. Do not choose a reference from
which camera sees it. This field does not certify current attachment, location,
or success; visible and target_uv must still be supported by the current image."""


class GroundedPolicy(VLMPolicy):
    # B10 paired-only observation regressed on genuine grasps. Kept for
    # reproducible static experiments, not enabled in the production pilot.
    paired_grasp_verification=False
    trackable_grasp_anchor=False

    def resolve_target_reference(self,harness):
        from .target_reference import ReferenceChoice,REFERENCES,REFERENCE_SYSTEM,reference_context
        from .vision import VisualBundle
        if not harness.held_inspection_enabled or not harness.reference_from_planner:
            raise ValueError("Separate semantic reference mode required")
        if harness.index in harness.target_references:
            ref=harness.search_reference
            if ref.startswith("held_") and not harness.hold_verified[ref[5:]]:
                harness.stop_reason="TARGET_REFERENCE_REQUIRES_VERIFIED_HOLD"
            return None
        context=reference_context(harness)
        if not any(harness.hold_verified.values()):
            harness.bind_reference("world","no_verified_held_reference_available")
            return None
        if "reference" not in self.identity.get("finite_choice_kinds",[]):
            raise ValueError("Service lacks constrained text-only semantic routing")
        used=getattr(self,"reference_calls",0)
        if used>=4:
            harness.stop_reason="SEMANTIC_REFERENCE_CALL_BUDGET_REACHED"
            return None
        self.reference_calls=used+1
        choices=tuple(ReferenceChoice(ref) for ref in REFERENCES)
        text=json.dumps(context,ensure_ascii=False)+"\nChoices:\n"+"\n".join(c.text() for c in choices)
        result,payload=self._call("reference",REFERENCE_SYSTEM,text,VisualBundle([],[],{},{}),choices)
        choice=ReferenceChoice.parse(result["text"])
        harness.bind_reference(choice.target_reference,"text_only_semantic_planner")
        return {"result":result,"request":payload}

    def observe(self,harness,state,bundle):
        bundle,instruction=verification_inputs(harness,bundle) if self.paired_grasp_verification else (bundle,"")
        text=perception_context(harness,state,bundle)
        if instruction:
            context=json.loads(text)
            context["current_robot"].pop("projection_guides",None)
            text=json.dumps(context,ensure_ascii=False)
        bimanual=harness.goal.kind=="pick" and harness.goal.hand=="both"
        system=BIMANUAL_OBSERVE_SYSTEM if bimanual else GROUNDED_OBSERVE_SYSTEM
        if not bimanual and not getattr(harness,"contact_geometry",True):system=GROUNDED_OBSERVE_CORE
        if getattr(harness,"held_inspection_enabled",False) and not harness.reference_from_planner:system=reference_observation_system(system)
        if self.trackable_grasp_anchor:instruction+=grasp_tracking_instruction(harness)
        result,payload=self._call("observe",system+instruction,text,bundle)
        evidence = (BimanualEvidence if bimanual else GroundedEvidence).parse(result["text"])
        if evidence.other_views:
            raise ValueError("Primary-contact interface requires other_views=[]; no guessed metric correspondences")
        validation = {"defaulted_fields": [] if "other_views" in strict_json(result["text"]) else ["other_views"],
                      "missing_other_views_means": "NO_CORROBORATING_EVIDENCE",
                      "raw_model_text_unchanged": True, "retries": 0,
                      "contact_contract":"one_view_per_contact; other-camera geometry from calibrated reprojection"}
        if getattr(harness,"held_inspection_enabled",False) and "target_reference" not in strict_json(result["text"]):
            validation["defaulted_fields"].append("target_reference")
        if bimanual and "hand_contacts" not in strict_json(result["text"]):
            validation["defaulted_fields"].append("hand_contacts")
            validation["missing_hand_contacts_means"]="NO_CONTACT_EVIDENCE; no shared-point fallback"
        return evidence,{"result":result,"request":payload,"validation":validation}

    def act_feasible(self,harness,state,bundle,allowed):
        if not allowed:
            raise ValueError("No preflighted action to select")
        text=actor_context(harness,state,bundle,allowed)
        text+="\nChoose one feasible command below; indices bind the scores, output ONLY its JSON:\n"+"\n".join(f"{index}: {a.text()}" for index,a in enumerate(allowed))
        system=ACTION_SYSTEM+" Prefer measurable progress toward the grounded contact region; review predicted distance gains and failed paths. For navigation, follow the navigation receipt: face the visible destination before approaching it; hand-to-surface distance is NOT a navigation completion test. The 2mm option remains available near limits. Do not repeatedly HOLD with good depth and a safe improving action. Grasp orientation/contact quality still require the raw views; a surface point is not a full grasp pose."
        if getattr(harness,"approach_reorientation",False):
            system += " During unloaded APPROACH, a fixed wrist pose can block reaching. When coarse translations are blocked, reorientation_after predicts whether a currently feasible wrist rotation could unlock a subsequent coarse reach. Consider such a pose change instead of indefinitely inching toward a joint limit, even when the rotation itself has near-zero distance gain. These two-command previews assume a stationary observed surface and certify only robot kinematics/self-collision, NOT environment clearance, visibility, grasp orientation or success. Choose ONLY the currently offered rotation, never its predicted followup; the next observation and safety checks must decide the next command. Use raw images to reject an unsafe-looking rotation."
        if getattr(harness,"approach_body_options",False):
            system += " An unloaded far-target approach may also offer base turns and alternate translations. All have current depth and robot preflight checks, not full scene collision certification. If repeated arm advances approach a joint limit, consider a feasible body reposition using the raw images and measured outcomes; do not assume the straight-ahead option is the only route. Base fine/micro translations are 6/2cm, turns 3/1 degrees. Predicted contact-distance gain assumes a stationary visible surface and is not a grasp-pose score or success. Choose one action only and observe its actual effect before another."
        if getattr(harness,"workspace_posture",False):
            system += " When moderate arm reaches are blocked, workspace_posture_after may describe an offered torso posture change. Torso fine moves its torso reference1cm while the coupled controller tries to KEEP BOTH HAND POSES FIXED; it does NOT mean moving both hands1cm. Such a first move can create arm workspace despite almost zero immediate hand-distance gain. Its following reach is only a robot-kinematic prediction, not environment clearance or contact safety. Inspect the raw views; select ONLY one currently offered command, then reobserve and recheck. Never execute the suggested followup as a queued plan or treat calibrated open fingers as proof of no contact."
        if getattr(harness,"contact_geometry",True):
            system += " For tool-frame TRANSLATIONS, forward/back are +/- local X, left/right +/- local Y, up/down +/- local Z (axis labels, not camera directions); use target_minus_center_tool_m to choose signs. Robot-only yellow contact strips and their polygon show the calibrated finger region at the current open aperture, not a target detection. An object close to the center but outside this region still needs alignment. If the region is visibly empty/below the target, align before CLOSE even when an exploratory close is offered; do not infer enclosure from the 4cm attempt bound. Choose a graspable narrow part, and change pose/approach if advancing only displaces the object."
        if harness.grasp_probe.get("eligible"):
            system += " A bounded active grasp probe is currently offered: CLOSE attempts a grasp but proves nothing. When the target is already near the open fingers and further advances keep pushing it, prefer a close attempt over continued pushing. You need not claim enclosure or holding before attempting; current enclosure is UNKNOWN, not verified true. Subsequent lift and independent evidence decide the outcome. You may still HOLD if the raw views contradict a safe attempt."
        if getattr(harness,"held_inspection_enabled",False) and harness.search_reference.startswith("held_") and harness.stage in ("SEARCH","RECOVER"):
            if getattr(harness,"multicamera_inspection",False):
                system += " The missing affordance belongs to an already held object. This is INSPECTION, not pressing yet. Target reference hand, moving hand, and observing camera are DISTINCT. You can move the confirmed-free hand to aim its wrist camera at the held object while holding the loaded arm and its closed grip fixed, or present the held object toward the head camera with the holding hand. Candidate inspection_after names its observer_camera and predicts a PREVIOUSLY observed surface anchor, NOT the button. Camera pointing_gain_deg improves framing; side_separation_from_head_deg distinguishes viewpoint direction. Rotating an attached camera with the held object does not reveal a different side. A free wrist initially pointing away may require several safe coarse rotations before its anchor enters view. Prefer useful pointing gains toward an independent view instead of repeatedly centering the same head view. Once framed, inspect CURRENT raw images for the affordance. Do not claim visibility or task completion from geometry. All observers share 24 attempts; holding and free arms have separate finite path/rotation accounts. Holding level/grip constraints remain active. Free-arm visible-depth sweeps only veto sampled visible obstacles, NOT unseen space or held-object clearance. HOLD if raw images contradict safety."
            else:
                system += " The missing affordance belongs to an already held object. This is INSPECTION, not pressing yet. Use the reference holding hand to present/reorient the object toward the head camera; rotating the base or the attached wrist camera together with the object does not reveal its hidden side. The free working hand is for the later interaction. Candidate inspection_after predicts only a previously observed surface anchor in the head image, NOT the button. Prefer bringing the object away from the head-image border toward its centre, then exposing another side with a permitted small rotation. Keep the grip closed; do not assume an affordance has appeared until the next actual observation. Respect level-preserving constraints and the finite inspection path budget."
            if harness.inspection_budget_aware and not getattr(harness,"multicamera_inspection",False):
                system += " Budget-aware inspection: exact image centering is NOT required and must not consume all 24 observations. When framing a border object, prefer an offered 3cm translation with a useful predicted framing gain over many 1cm repeats, unless raw images suggest danger. Once the object can be inspected, spend observations on new angular views to expose another side; do not wait for perfect centering. Previously visited head-to-hand poses are excluded, so do not alternate a rotation and its undo. A new pose is not proof that the button is visible. Kinematic preflight covers only robot joint limits and self-collision; environment and held-object collision safety are NOT certified. HOLD if the visible scene contradicts a candidate."
        result,payload=self._call("act",system,text,bundle,allowed)
        action=Action.parse(result["text"])
        harness.authorize(action)
        if action not in allowed:
            raise ValueError("Action not in current-state preflight results")
        return action,{"result":result,"request":payload}

    def recover(self,harness,state,bundle,reason):
        text=actor_context(harness,state,bundle)+"\nRecovery trigger: "+reason
        options = {"response_schema": "recovery_v1"} if self.structured_planning else {}
        result,payload=self._call("plan",RECOVER_SYSTEM,text,bundle,**options)
        return parse_recovery(result["text"]),{"result":result,"request":payload}


SURFACE_SYSTEM = """Choose a visible contact surface for the CURRENT target, not a robot action. Full CURRENT raw views, a separate overview marking the crop's location, and an enlarged raw/numbered crop are supplied. The cyan rectangle locates the crop: it is NOT an object detection. Cyan numbered dots are depth-stable surface CANDIDATES, NOT object detections. Some dots may lie on background or a different object. Use the full view to identify which object the patch belongs to; a component need not share the color in the object's name. Check the unmarked crop and then its numbered counterpart. For pick prefer a clearly visible surface near the intended graspable region; this surface is only a position reference, not a complete grasp pose. For press select its actual button, for open/close its handle, for navigate the named destination. Select one candidate_id only if its dot is visibly ON the requested object/affordance; otherwise choose null. Never choose merely because a point is nearest or has small depth. Output exactly one listed JSON choice. No motion, enclosure, holding, completion or cross-view correspondence is certified."""


class RefinedGroundedPolicy(GroundedPolicy):
    def __init__(self,*args,max_refinements=16,**kwargs):
        super().__init__(*args,**kwargs)
        if "ground" not in self.identity.get("finite_choice_kinds",[]):
            raise ValueError("Service must support constrained surface choices before a refined run")
        if type(max_refinements) is not int or not 1<=max_refinements<=16:
            raise ValueError("Bounded refinement budget required")
        self.max_refinements,self.refinements=max_refinements,0

    def refine(self,evidence,harness,state,bundle,depths,model):
        if harness.goal.kind=="pick" and harness.goal.hand=="both":
            # At most ONE extra model choice per decision, shared 16-call cap.
            # Refine a supplied per-hand ray only; never manufacture a missing
            # contact or substitute the global object centroid for both hands.
            receipt={"attempted":False,"original_evidence":asdict(evidence),
                     "reason":"BIMANUAL_CONTACTS_VALID_OR_UNKNOWN","robot_controls":0,
                     "refinements_used":self.refinements}
            for contact in getattr(evidence,"hand_contacts",()):
                single=contact_evidence(evidence,contact)
                if not single.visible or localize_target(single,depths,model,state.q)["valid"]:
                    continue
                from types import SimpleNamespace
                goal=replace(harness.goal,hand=contact.hand,
                             done_when="This hand and the other hand both support the same target after a lift.")
                scope=SimpleNamespace(goal=goal,stage=harness.stage)
                refined,call,detail,views=self.refine(single,scope,state,bundle,depths,model)
                updated=replace(contact,view=refined.view,target_uv=refined.target_uv,
                                enclosed=refined.enclosed,co_moving=refined.co_moving)
                result=replace(evidence,hand_contacts=tuple(updated if row.hand==contact.hand else row
                                                          for row in evidence.hand_contacts))
                receipt.update(attempted=detail["attempted"],reason=detail["reason"],
                               hand=contact.hand,contact_receipt=detail,refined_evidence=asdict(result),
                               refinements_used=self.refinements)
                return result,call,receipt,views
            return evidence,None,receipt,None
        target=localize_target(evidence,depths,model,state.q)
        receipt={"attempted":False,"original_target":target,"refinements_used":self.refinements,
                 "original_evidence":asdict(evidence),"robot_controls":0}
        if (not evidence.visible or target["valid"] or harness.stage in
                ("GRASP","VERIFY_GRASP","RELEASE","VERIFY_PLACE","VERIFY_SUPPORT","VERIFY_EFFECT")):
            receipt["reason"]="NOT_NEEDED_OR_VERIFICATION_STAGE"
            return evidence,None,receipt,None
        if self.refinements>=self.max_refinements:
            receipt["reason"]="REFINEMENT_BUDGET_EXHAUSTED"
            return evidence,None,receipt,None
        proposals=surface_candidates(evidence,depths,model,state.q)
        receipt["proposals"]=proposals
        if not proposals["candidates"]:
            receipt["reason"]="NO_STABLE_SURFACE_PROPOSALS"
            return evidence,None,receipt,None
        views=refinement_bundle(bundle,proposals)
        allowed=(SurfaceChoice(),*(SurfaceChoice(c["id"]) for c in proposals["candidates"]))
        text=json.dumps(surface_selection_context(harness.goal,harness.stage,proposals))
        text+="\nChoose exactly one:\n"+"\n".join(c.text() for c in allowed)
        self.refinements+=1  # even a failed call consumes the bounded allowance
        result,payload=self._call("ground",SURFACE_SYSTEM,text,views,allowed)
        choice=SurfaceChoice.parse(result["text"])
        if choice not in allowed:raise ValueError("Choice not proposed in the current snapshot")
        refined=select_surface(evidence,choice,proposals)
        receipt.update(attempted=True,refinements_used=self.refinements,choice=asdict(choice),
                       reason="MODEL_ABSTAINED" if choice.candidate_id is None else "MODEL_SELECTED_CURRENT_SURFACE",
                       refined_evidence=asdict(refined),refined_target=localize_target(refined,depths,model,state.q))
        return refined,{"result":result,"request":payload},receipt,views

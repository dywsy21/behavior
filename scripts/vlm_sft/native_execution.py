"""Explicit execution semantics, separate from every physical success gate.

Missing profile preserves legacy behavior. New receipts are never accepted
by a status-name whitelist, and neither path certifies a grasp or release.
"""
from native_motion_codec import (token_to_action, VERSION as WORKSPACE_CODEC,
    BODY_TOKENS, protocol_for_codec)
import numpy as np
from native_actor_protocol import finite
from semantic_robot.v2.protocol import TRANSLATIONS
from semantic_robot.v2.servo import (ServoLimits, execution_completed,
    GRIPPER_COMPLETION_VERSION, GRIPPER_COMPLETED)

PROFILE = GRIPPER_COMPLETION_VERSION
PRECLOSE_PROFILE = "gripper-command-completion-v1-public-preclose-translation-v1"
WORKSPACE_PROFILE = PRECLOSE_PROFILE + "-compensated-torso-v1"
CARRY_PROFILE = WORKSPACE_PROFILE + "-carry-duration-v1-body2"
WORKSPACE_PROFILES = (WORKSPACE_PROFILE, CARRY_PROFILE)
TIMING_PROFILES = (PRECLOSE_PROFILE, *WORKSPACE_PROFILES)
CARRY_CONTRACT = {"carry_duration_v1": True, "max_workspace_macros": 2,
                  "max_decisions": 14, "new_controls_including_final_hold": 640}


def episode_limits(profile):
    validate_profile(profile)
    return (14, 640) if profile == CARRY_PROFILE else (12, 420)


def service_call_limit(profile):
    return 4*episode_limits(profile)[0]


def body_limit(profile):
    validate_profile(profile)
    return 2 if profile == CARRY_PROFILE else int(profile == WORKSPACE_PROFILE)


def primitive_tick_limit(action, carry, profile):
    validate_profile(profile)
    old = {"micro": 24, "fine": 40, "coarse": 64}[action.scale]
    return servo_limits(profile).action_tick_limit(action,carry) if profile==CARRY_PROFILE else old


def validate_profile(value):
    if value is not None and value not in (PROFILE, *TIMING_PROFILES):
        raise ValueError("Unknown execution profile")
    return value


def authorization_profile(value, *, collection=False):
    if "execution_profile" not in value:
        if "action_codec" in value or any(k in value for k in CARRY_CONTRACT):raise ValueError("Codec/budget without execution profile")
        return None
    profile = value["execution_profile"]
    if profile not in (PROFILE, *TIMING_PROFILES):
        raise ValueError("Explicit execution profile must be named, not null/implicit")
    if profile in WORKSPACE_PROFILES and value.get("action_codec") != WORKSPACE_CODEC:
        raise ValueError("Explicit compensated-torso codec binding required")
    if profile not in WORKSPACE_PROFILES and "action_codec" in value:
        raise ValueError("Legacy execution cannot acquire a new codec implicitly")
    if profile == CARRY_PROFILE:
        if any(type(value.get(k)) is not type(v) or value[k] != v for k,v in CARRY_CONTRACT.items()):
            raise ValueError("Exact explicit carry-duration/body2/14-macro/640-control contract required")
    elif any(k in value for k in CARRY_CONTRACT):
        raise ValueError("Old execution profiles cannot acquire new duration or episode limits")
    if collection:
        from native_teacher_capacity import (H09Y_PROFILE, DIVERSE_PROFILE, WORKSPACE_PROFILE as WORKSPACE_CAPACITY,
                                             CARRY_PROFILE as CARRY_CAPACITY)
        from native_teacher_near_grasp import SCHEMA
        from native_teacher_pregrasp_seed import PROFILE as SEED_PROFILE
        capacity = {PRECLOSE_PROFILE:DIVERSE_PROFILE,WORKSPACE_PROFILE:WORKSPACE_CAPACITY,
                    CARRY_PROFILE:CARRY_CAPACITY}.get(profile,H09Y_PROFILE)
        if (value.get("schema") != SCHEMA or value.get("capacity_profile") != capacity or
                value.get("seed_profile") != SEED_PROFILE):
            raise ValueError("New gripper collection requires explicit H09Y precontact-v2")
        if profile in TIMING_PROFILES:
            from native_storage import (DIVERSE_STORAGE_PROFILE, WORKSPACE_STORAGE_PROFILE, CARRY_STORAGE_PROFILE,
                                        CARRY953_STORAGE_PROFILE,validate_spec)
            from native_teacher_capacity import capacity_limits,CARRY_ADDITIONAL_START
            capacity_limits(value)
            storage = {WORKSPACE_PROFILE:WORKSPACE_STORAGE_PROFILE,CARRY_PROFILE:CARRY_STORAGE_PROFILE}.get(profile,DIVERSE_STORAGE_PROFILE)
            if profile==CARRY_PROFILE and (*value['source'],value['paid_prefix_controls'])==CARRY_ADDITIONAL_START:
                storage=CARRY953_STORAGE_PROFILE
            if not validate_spec(value) or value["storage"]["profile"] != storage:
                raise ValueError("Diverse collection requires its exact explicit storage profile")
    return profile


def metadata(profile):
    validate_profile(profile)
    result = {} if profile is None else {"execution_profile": profile}
    if profile in WORKSPACE_PROFILES: result["action_codec"] = WORKSPACE_CODEC
    if profile == CARRY_PROFILE: result.update(CARRY_CONTRACT)
    return result


def action_codec(profile):
    return WORKSPACE_CODEC if validate_profile(profile) in WORKSPACE_PROFILES else None


def actor_protocol(profile):
    return protocol_for_codec(action_codec(profile))


def servo_limits(profile):
    return ServoLimits(robot_geometry_guards=True,
                       gripper_completion_v1=validate_profile(profile) is not None,
                       **({"carry_duration_v1": True} if profile == CARRY_PROFILE else {}))


def require_dataset_profile(dataset,profile):
    # Historical successful W remains legitimate supervision, but a dataset
    # containing new-profile trajectories cannot silently evaluate old servo.
    profiles={authorization_profile(run) for run in dataset["runs"]}
    expected=next((p for p in (CARRY_PROFILE,WORKSPACE_PROFILE,PRECLOSE_PROFILE,PROFILE) if p in profiles),None)
    if validate_profile(profile)!=expected:
        raise ValueError("Paired execution profile differs from the admitted TRAIN data")


def completed(token, feedback, *, profile=None):
    """Bind a receipt to the actual token and the explicitly enabled profile."""
    validate_profile(profile)
    action = token_to_action(token, action_codec(profile))
    if not isinstance(feedback, dict):
        return False
    gripper = action.move in ("open", "close")
    if profile == CARRY_PROFILE:
        ticks=feedback.get("control_ticks")
        if (type(feedback.get("carry")) is not bool or type(ticks) is not int or
                not 1 <= ticks <= primitive_tick_limit(action,feedback["carry"],profile)):
            return False
        m=feedback.get("motion_timing")
        from semantic_robot.v2.protocol import ROTATIONS
        planned=action.part!='base' and action.move in (*TRANSLATIONS,*ROTATIONS)
        expected={"version":"carry_duration_v1","maximum_control_ticks":primitive_tick_limit(action,feedback['carry'],profile),
            "required_joint_trajectory_ticks":ticks if planned else None,"joint_trajectory_planned":planned,
            "joint_step_limit":.0125 if feedback['carry'] else .025,"joint_trajectory_settling_ticks":5,"success_claim":False}
        if not isinstance(m,dict) or set(m)!=set(expected) or any(type(m[k]) is not type(v) or m[k]!=v for k,v in expected.items()):return False
    elif 'motion_timing' in feedback:return False
    if token in BODY_TOKENS:
        try:
            if (feedback.get("carry") is not False or type(feedback.get("control_ticks")) is not int or
                    not 18 <= feedback["control_ticks"] <= 40 or feedback.get("joint_limit_ticks") != 0 or
                    feedback.get("holding") != "UNKNOWN" or feedback.get("official_success") != "NOT_AVAILABLE_TO_ACTOR"):
                return False
            for name in ("left","right","torso"):
                p=float(feedback["target_error_m"][name]);a=float(feedback["orientation_error_deg"][name])
                if not np.isfinite([p,a]).all() or not (0 <= p <= .0025 and 0 <= a <= 1.5+1e-10):return False
        except (KeyError,TypeError,ValueError):return False
    if profile is not None and gripper:
        # New enabled servo always emits the detailed gripper receipt. Do not
        # silently accept a legacy/mixed implementation under the new profile.
        return feedback.get("status") == GRIPPER_COMPLETED and execution_completed(action, feedback)
    if "gripper_execution" in feedback or "pose_tracking_status" in feedback:
        return False
    return feedback.get("status") == "TARGET_REACHED" and execution_completed(action, feedback)


class PublicGripperHistory:
    """Local-pause command latch, never contact/holding truth or a resettable token history."""
    def __init__(self, grips):
        self.grips = finite(grips, (2,)).copy()
        if np.any(abs(self.grips) > 1):
            raise ValueError("Native gripper command range")
        self.close_seen = {a: bool(self.grips[i] < .999) for i, a in enumerate(("left", "right"))}

    def issued(self, command):
        command = finite(command, (23,)); g = command[[14, 22]]
        if np.any(abs(g) > 1):
            raise ValueError("Native gripper command range")
        self.grips = g.copy()
        for i, arm in enumerate(("left", "right")):
            if g[i] < .999:
                self.close_seen[arm] = True

    def rotation_qualified(self, arm, state, model):
        # This same robot-only full-open qualification is also used for timing.
        # It does NOT certify an empty hand or absence of environmental contact.
        if arm not in ("left", "right") or self.close_seen.get(arm) is not False:
            return False
        try:
            i = ("left", "right").index(arm); data = model.spec["metadata"]
            reference = finite(data["grasp_region_reference_gripper_m"], (2,))
            actual = finite(state.gripper, (2,))
            return bool(.999 <= self.grips[i] <= 1 and
                data["grasp_region_reference_fully_open"].get(arm) is True and
                actual[i] >= .0495 and abs(actual[i] - reference[i]) <= .0005)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


class WorkspaceQuota:
    """Count a macro once at its first actual command, never on proposal/completion."""
    def __init__(self, profile):
        self.limit=body_limit(profile);self.count=0;self.pending=False

    def available(self):return self.count < self.limit

    def arm(self, token):
        self.pending=False
        if token in BODY_TOKENS:
            if not self.available():raise RuntimeError("Workspace macro quota exhausted")
            self.pending=True

    def issued(self, command):
        command=finite(command,(23,))
        if np.any(abs(command[[14,22]])>1):raise ValueError("Native gripper command range")
        if self.pending:
            if not self.available():raise RuntimeError("Workspace macro quota exhausted")
            self.count+=1;self.pending=False

    def cancel_pending(self):self.pending=False


def preclose_translation(action, state, model, history, profile):
    """Select original normal timing for exactly qualified fine arm translations."""
    validate_profile(profile)
    return bool(profile in TIMING_PROFILES and isinstance(history, PublicGripperHistory) and
        action.part in ("left", "right") and action.move in TRANSLATIONS and
        action.scale == "fine" and action.frame == "base" and
        history.rotation_qualified(action.part, state, model))


def timing_metadata(profile, qualified):
    return {} if profile not in TIMING_PROFILES else {
        "public_preclose_translation_qualification": bool(qualified),
        "empty_hand_or_environment_contact_not_certified": True}


def workspace_qualified(action,state,model,history,profile):
    return bool(validate_profile(profile) in WORKSPACE_PROFILES and isinstance(history,PublicGripperHistory) and
        action.part=="torso" and action.move in ("up","down","forward","back") and
        action.scale=="fine" and action.frame=="base" and
        all(history.rotation_qualified(arm,state,model) for arm in ("left","right")))


def workspace_metadata(action,qualified,profile):
    if profile not in WORKSPACE_PROFILES:return {}
    return {"public_both_open_no_close_workspace_qualification":bool(qualified),
        "compensated_torso_keep_both_eef":action.part=="torso",
        "scene_sweep_clearance_certified":False}


def workspace_budget_ok(token,planned_ticks,remaining_controls,remaining_macros,profile=WORKSPACE_PROFILE):
    if token not in BODY_TOKENS:return True
    # Same public worst-case next fine primitive in collector and all policies.
    return bool(all(type(v) is int for v in (planned_ticks,remaining_controls,remaining_macros)) and
        18<=planned_ticks<=40 and remaining_macros>=2 and
        remaining_controls>=planned_ticks+12+(75 if profile==CARRY_PROFILE else 40)+12+1)


def require_pipeline_profile(release, dataset):
    """New timing/storage must be identical for TRAIN, service and all three actors."""
    profile = authorization_profile(release)
    if (profile in WORKSPACE_PROFILES or dataset.get("protocol")==actor_protocol(WORKSPACE_PROFILE) or
            any(authorization_profile(r) in WORKSPACE_PROFILES for r in dataset["runs"])):
        from native_storage import WORKSPACE_STORAGE_PROFILE,CARRY_STORAGE_PROFILES,CARRY953_STORAGE_PROFILE,validate_spec
        storage_profiles=CARRY_STORAGE_PROFILES if profile==CARRY_PROFILE else (WORKSPACE_STORAGE_PROFILE,)
        if any(run.get('group')==[1,114] and run.get('prefix')==953 for run in dataset['runs']):
            storage_profiles=(CARRY953_STORAGE_PROFILE,)
        require_dataset_profile(dataset,profile)
        if (profile not in WORKSPACE_PROFILES or release.get("protocol")!=actor_protocol(profile) or
                dataset.get("protocol")!=actor_protocol(profile) or not validate_spec(release) or
                release["storage"]["profile"] not in storage_profiles):
            raise ValueError("Workspace pipeline requires exact new codec/protocol/storage")
        if profile==CARRY_PROFILE:
            from native_carry_admission import require_carry_reviews
            require_carry_reviews(release)
        return profile
    new_data = any(authorization_profile(run) == PRECLOSE_PROFILE for run in dataset["runs"])
    from native_storage import DIVERSE_STORAGE_PROFILE, validate_spec
    new_storage = (release.get("storage") or {}).get("profile") == DIVERSE_STORAGE_PROFILE
    if new_data or profile == PRECLOSE_PROFILE or new_storage:
        require_dataset_profile(dataset, profile)
        if (profile != PRECLOSE_PROFILE or not validate_spec(release) or
                release["storage"]["profile"] != DIVERSE_STORAGE_PROFILE):
            raise ValueError("Public preclose pipeline requires the exact diverse storage profile")
    return profile


def require_same_pipeline(release, identity):
    profile = authorization_profile(release)
    if profile in WORKSPACE_PROFILES or authorization_profile(identity) in WORKSPACE_PROFILES:
        if (authorization_profile(identity)!=profile or identity.get("protocol")!=actor_protocol(profile) or
                release.get("protocol")!=actor_protocol(profile) or
                (identity.get("storage") or {}).get("profile")!=(release.get("storage") or {}).get("profile")):
            raise ValueError("Workspace training/service/evaluation identity differs")
        if profile==CARRY_PROFILE and any(not release.get(k) or release[k]!=identity.get(k)
                for k in ('code_commit','executor_digest','carry_duration_core_commit')):
            raise ValueError("Carry pipeline exact immutable source/core/executor differs")
        return
    if profile == PRECLOSE_PROFILE or authorization_profile(identity) == PRECLOSE_PROFILE:
        if (authorization_profile(identity) != profile or
                (identity.get("storage") or {}).get("profile") != (release.get("storage") or {}).get("profile")):
            raise ValueError("Training/service/evaluation execution or storage profile differs")

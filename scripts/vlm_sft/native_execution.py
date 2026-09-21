"""Explicit execution semantics, separate from every physical success gate.

Missing profile preserves legacy behavior. New receipts are never accepted
by a status-name whitelist, and neither path certifies a grasp or release.
"""
from common import token_to_action
from semantic_robot.v2.servo import (ServoLimits, execution_completed,
    GRIPPER_COMPLETION_VERSION, GRIPPER_COMPLETED)

PROFILE = GRIPPER_COMPLETION_VERSION


def validate_profile(value):
    if value is not None and value != PROFILE:
        raise ValueError("Unknown execution profile")
    return value


def authorization_profile(value, *, collection=False):
    if "execution_profile" not in value:
        return None
    profile = value["execution_profile"]
    if profile != PROFILE:
        raise ValueError("Explicit execution profile must be named, not null/implicit")
    if collection:
        from native_teacher_capacity import H09Y_PROFILE
        from native_teacher_near_grasp import SCHEMA
        from native_teacher_pregrasp_seed import PROFILE as SEED_PROFILE
        if (value.get("schema") != SCHEMA or value.get("capacity_profile") != H09Y_PROFILE or
                value.get("seed_profile") != SEED_PROFILE):
            raise ValueError("New gripper collection requires explicit H09Y precontact-v2")
    return profile


def metadata(profile):
    validate_profile(profile)
    return {} if profile is None else {"execution_profile": profile}


def servo_limits(profile):
    return ServoLimits(robot_geometry_guards=True,
                       gripper_completion_v1=validate_profile(profile) == PROFILE)


def require_dataset_profile(dataset,profile):
    # Historical successful W remains legitimate supervision, but a dataset
    # containing new-profile trajectories cannot silently evaluate old servo.
    profiles={authorization_profile(run) for run in dataset["runs"]}
    expected=PROFILE if PROFILE in profiles else None
    if validate_profile(profile)!=expected:
        raise ValueError("Paired execution profile differs from the admitted TRAIN data")


def completed(token, feedback, *, profile=None):
    """Bind a receipt to the actual token and the explicitly enabled profile."""
    validate_profile(profile)
    action = token_to_action(token)
    if not isinstance(feedback, dict):
        return False
    gripper = action.move in ("open", "close")
    if profile == PROFILE and gripper:
        # New enabled servo always emits the detailed gripper receipt. Do not
        # silently accept a legacy/mixed implementation under the new profile.
        return feedback.get("status") == GRIPPER_COMPLETED and execution_completed(action, feedback)
    if "gripper_execution" in feedback or "pose_tracking_status" in feedback:
        return False
    return feedback.get("status") == "TARGET_REACHED" and execution_completed(action, feedback)

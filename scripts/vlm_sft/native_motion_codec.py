"""Explicit native-only compensated torso symbols; legacy expert codec unchanged."""
from common import TOKENS as LEGACY_TOKENS, token_to_action as legacy_action

VERSION = "h09z-native-keep-eef-45-v1"
ACTOR_VERSION = "h09z-current-robot-pose-workspace-v1"
LEGACY_ACTOR_VERSION = "h09x-current-robot-pose-v1"
BODY_TOKENS = tuple(f"TORSO_{move}_KEEP_EEF" for move in ("UP", "DOWN", "FORWARD", "BACK"))
TOKENS = LEGACY_TOKENS + BODY_TOKENS
BODY_SYSTEM = " TORSO_*_KEEP_EEF moves the torso link 1 cm along the named robot BASE axis while compensating both arms to preserve both end-effector poses. It is not an arm lift or a grasp claim. These motions require both grippers currently calibrated fully open and neither hand commanded CLOSE since the local pause."


def tokens(codec=None):
    if codec is None: return LEGACY_TOKENS
    if codec != VERSION: raise ValueError("Unknown explicit native motion codec")
    return TOKENS


def codec_for_protocol(protocol):
    if protocol == LEGACY_ACTOR_VERSION: return None
    if protocol == ACTOR_VERSION: return VERSION
    raise ValueError("Unknown current-robot action protocol")


def protocol_for_codec(codec):
    tokens(codec)
    return LEGACY_ACTOR_VERSION if codec is None else ACTOR_VERSION


def token_to_action(token, codec=None):
    if token not in tokens(codec): raise ValueError("Token outside explicitly selected native codec")
    if token in BODY_TOKENS:
        from semantic_robot.v2.protocol import Action
        return Action("torso", token.split("_")[1].lower(), "fine", "base")
    return legacy_action(token)

"""Privileged OFFLINE proposals, not action-correctness labels.

Goal seeds are explicitly reviewed object-local poses, not a hand-written
trajectory. The same geometry/codec loop serves every task. Missing seeds
stop BEFORE reset; failed terminal predicates never release positive BC.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from common import TOKENS, token_to_action
from native_teacher_outcomes import rigid
from semantic_robot.v2.protocol import TRANSLATIONS, ROTATIONS


def validate_spec(spec, reference):
    allowed = {"schema", "verb", "hand", "support_hand", "target", "destination", "payloads",
               "goal_pose_local", "goal_frame", "pose_reviewer", "pose_evidence_sha256"}
    if set(spec) != allowed or spec["schema"] != "h09t-private-teacher-v1":
        raise ValueError("Exact private teacher schema required")
    import json
    skills = json.loads(reference["private_original_semantic_json"])
    if len(skills) != 1:
        raise ValueError("One same-state source skill only; no mixed/bundled labels")
    source = skills[0]
    if (source.get("unbound_relation") or spec["verb"] != source["verb"] or
            spec["target"] != source["target"] or spec["destination"] != source["destination"] or
            spec["hand"] not in ("left", "right") or
            spec["support_hand"] not in (None, "left", "right") or
            spec["support_hand"] == spec["hand"] or not isinstance(spec["payloads"], list) or
            len(set(spec["payloads"])) != len(spec["payloads"])):
        raise ValueError("Teacher/source intent, identity or hand binding mismatch")
    if source["arm"] not in ("UNSPECIFIED", spec["hand"].upper()):
        raise ValueError("Teacher hand contradicts source skill")
    verb = spec["verb"]
    expected = "target" if verb == "GRASP" else "toggle_link" if verb == "PRESS" else "destination"
    if verb not in ("GRASP", "PRESS", "PLACE_IN", "PLACE_ON") or spec["goal_frame"] != expected:
        raise ValueError("Unsupported or unbound teacher goal frame")
    rigid(spec["goal_pose_local"])
    if (not spec["pose_reviewer"] or not isinstance(spec["pose_evidence_sha256"], str) or
            len(spec["pose_evidence_sha256"]) != 64 or
            any(c not in "0123456789abcdef" for c in spec["pose_evidence_sha256"])):
        raise ValueError("Reviewed object-local pose/affordance seed required, not default object center")


def validate_train_group(reference, counts, held_out_groups):
    source = reference["source"]
    group = (source["task"], source["instance"])
    excluded = {tuple(x) for rows in counts["exclusions"].values() for x in rows}
    excluded |= {tuple(x) for x in held_out_groups}
    excluded |= {(0, 138), (3, 242)}
    matches = [r for r in counts["sources"] if (r["task"], r["episode"], r["instance"]) ==
               (source["task"], source["episode"], source["instance"])]
    if group in excluded or len(matches) != 1 or source["cohort"] != "additional_train":
        raise ValueError("Protected/held-out/ambiguous source instance, never TRAIN")
    if matches[0].get("extracted_arrays_and_labels_sha256") != source["extracted_arrays_and_labels_sha256"]:
        raise ValueError("TRAIN source arrays/labels identity mismatch")
    return group


def empty_hand_rotation_allowed(arm, state, frame, grips, model, close_issued):
    """OFFLINE evidence only; aperture alone never proves an empty hand."""
    if arm not in ("left","right") or close_issued is not False or state is None or model is None:
        return False
    i=("left","right").index(arm)
    try:
        g=np.asarray(grips,float);actual=np.asarray(state.gripper,float)
        metadata=model.spec["metadata"]
        ref=np.asarray(metadata["grasp_region_reference_gripper_m"],float)
        return bool(g.shape==(2,) and actual.shape==(2,) and ref.shape==(2,) and
                    np.isfinite(np.r_[g,actual,ref]).all() and .999<=g[i]<=1 and
                    metadata["grasp_region_reference_fully_open"].get(arm) is True and
                    abs(actual[i]-ref[i])<=.0005 and actual[i]>=.0495 and
                    frame.get("contacts_known") is True and
                    frame.get("held_any",{}).get(arm) is False and
                    frame.get("held",{}).get(arm) is False and
                    frame.get("finger_external_contact",{}).get(arm) is False and
                    frame.get("finger_contact",{}).get(arm) is False)
    except (KeyError,TypeError,ValueError):return False


class PoseTeacher:
    def __init__(self, spec, allow_empty_rotation=False):
        self.spec = spec
        self.allow_empty_rotation = allow_empty_rotation is True
        self.stage = "APPROACH"
        self.close_issued = self.open_issued = False
        self.lift_origin = None
        self.visits = {}

    def ranked(self, state, frame, goal_world, base_world, grips=None, model=None):
        arm, verb = self.spec["hand"], self.spec["verb"]
        if frame["held"][arm] is None:
            raise RuntimeError("UNKNOWN hold state; no teacher action")
        current = rigid(frame["hand_poses"][arm])
        goal = rigid(goal_world).copy()
        if verb == "GRASP" and self.close_issued:
            if frame["held"][arm] is not True or frame["finger_contact"][arm] is not True:
                raise RuntimeError("Close did not establish target-identity contact; no retry")
            if self.lift_origin is None: self.lift_origin = current.copy()
            goal = self.lift_origin.copy(); goal[2, 3] += .04
            self.stage = "LIFT_VERIFY"
        elif verb.startswith("PLACE"):
            if self.open_issued:
                self.stage = "RELEASE_VERIFY"
                return ["HOLD"]
            if frame["held"][arm] is not True:
                raise RuntimeError("Placement starts without a verified target hold")
            # Goal describes TARGET pose in destination coordinates; preserve
            # the actual current hand-to-object transform in this prediction.
            hand_object = np.linalg.inv(current)@rigid(frame["target_pose"])
            goal = goal@np.linalg.inv(hand_object)
        base = rigid(base_world)
        local_current, local_goal = np.linalg.inv(base)@current, np.linalg.inv(base)@goal
        distance = np.linalg.norm(local_current[:3, 3]-local_goal[:3, 3])
        angle = np.linalg.norm(Rotation.from_matrix(local_goal[:3, :3]@local_current[:3, :3].T).as_rotvec())
        if distance <= .004 and angle <= np.deg2rad(4):
            if verb == "GRASP" and not self.close_issued: return [arm.upper()+"_CLOSE"]
            if verb.startswith("PLACE") and not self.open_issued: return [arm.upper()+"_OPEN"]
            return ["HOLD"]
        candidates = []
        for token in TOKENS:
            if not token.startswith(arm.upper()+"_"): continue
            action = token_to_action(token)
            predicted = local_current.copy()
            if action.move in TRANSLATIONS:
                predicted[:3, 3] += np.asarray(TRANSLATIONS[action.move])*action.amount(True)
            elif action.move in ROTATIONS:
                if (not self.allow_empty_rotation or not empty_hand_rotation_allowed(
                        arm,state,frame,grips,model,self.close_issued)):continue
                # Existing token_to_action is BASE-frame fine rotation (3deg),
                # NOT a tool-axis reinterpretation. Translations stay carry=True.
                if action.frame!="base":raise ValueError("Unreviewed teacher action frame")
                predicted[:3,:3]=Rotation.from_rotvec(
                    np.asarray(ROTATIONS[action.move])*action.amount(False)).as_matrix()@predicted[:3,:3]
            else: continue
            cost = (np.linalg.norm(predicted[:3, 3]-local_goal[:3, 3]) +
                    .04*np.linalg.norm(Rotation.from_matrix(local_goal[:3, :3]@predicted[:3, :3].T).as_rotvec()))
            candidates.append((cost, token))
        current_cost = distance+.04*angle
        return [token for cost, token in sorted(candidates) if cost < current_cost-1e-6][:12]

    def executed(self, token, frame):
        if token == self.spec["hand"].upper()+"_CLOSE": self.close_issued = True
        if token == self.spec["hand"].upper()+"_OPEN": self.open_issued = True
        transform = rigid(frame["hand_poses"][self.spec["hand"]])
        key = tuple(np.round(np.r_[transform[:3, 3]/.003,
                        Rotation.from_matrix(transform[:3, :3]).as_rotvec()/np.deg2rad(1)]).astype(int))
        self.visits[key] = self.visits.get(key, 0)+1
        if self.visits[key] > 4:
            raise RuntimeError("Bounded teacher made no motion progress; no reset/retry")

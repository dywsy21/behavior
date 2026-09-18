"""Explicit, hash-pinned diagnostic action replay; never autonomous progress."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .harness import parse_plan
from .protocol import Action


def load_saved_prefix(path,*,task,prefix,window_sha,robot_sha):
    path=Path(path);spec=json.loads(path.read_text());root=Path(spec["source_run"])
    n=spec["replay_decisions"];count=spec["replay_controls"]
    approach=(spec.get("schema")==2 and spec.get("purpose")=="matched_unladen_approach_diagnostic")
    if (spec.get("schema") not in (1,2) or (spec.get("schema")==2 and not approach)
            or type(n) is not int or not 1<=n<=64 or type(count) is not int or not 1<=count<=(2048 if approach else 512)):
        raise ValueError("Bounded diagnostic replay required")
    required={"manifest.json","steps.jsonl","plan.json",f"decision_{n:03d}/proprio.json"}
    if approach:required.add(f"decision_{n:03d}/harness.json")
    if set(spec["sha256"])!=required:raise ValueError("Exact replay source identities required")
    payload={}
    for name in sorted(required):
        raw=(root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=spec["sha256"][name]:raise ValueError("Saved replay SHA mismatch: "+name)
        payload[name]=raw.decode()
    manifest=json.loads(payload["manifest.json"])
    if (manifest["task"]!=task or manifest["args"]["prefix"]!=prefix or manifest["window_sha"]!=window_sha or
            manifest["robot_sha"]!=robot_sha or manifest["code_commit"]!=spec["source_code_commit"]):
        raise ValueError("Replay task/window/robot/prefix/source differs")
    rows=[json.loads(line) for line in payload["steps.jsonl"].splitlines()]
    controls=[r for r in rows if "action23" in r and r["decision"]<n]
    commands=[r for r in rows if "action" in r and "feedback" in r and r["decision"]<n]
    if (len(controls)!=count or [r["control"] for r in controls]!=list(range(1,count+1)) or
            [r["decision"] for r in commands]!=list(range(n)) or commands[-1]["control_end"]!=count or
            sum(r["feedback"]["control_ticks"] for r in commands)!=count or any(r.get("terminal") for r in controls)):
        raise ValueError("Missing, out-of-order or terminated replay controls")
    actions=np.asarray([r["action23"] for r in controls],dtype=float)
    if actions.shape!=(count,23) or not np.isfinite(actions).all():raise ValueError("Finite real23 actions required")
    plan=parse_plan(payload["plan.json"]);last=Action(**commands[-1]["action"])
    expected=json.loads(payload[f"decision_{n:03d}/proprio.json"])
    q,g=np.asarray(expected["q"]),np.asarray(expected["gripper"])
    if q.shape!=(18,) or g.shape!=(2,) or not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Finite recorded replay endpoint required")
    if approach:
        h=json.loads(payload[f"decision_{n:03d}/harness.json"])
        index=h.get("goal_index")
        if (prefix!=0 or manifest["args"].get("replay_prefix_spec") is not None or
                type(index) is not int or not 0<=index<len(plan) or h.get("goal")!=asdict(plan[index]) or
                plan[index].kind!="pick" or plan[index].hand not in ("left","right") or
                h.get("stage")!="APPROACH" or h.get("stop_reason") is not None or
                h.get("carry_constraints") is not False or np.any(g<.0495) or
                not np.array_equal(actions[-1,[14,22]],[1,1])):
            raise ValueError("Only original-source unloaded single-hand APPROACH may be replayed")
        for field in ("unverified_close_latches","holding_verified_by_observation_and_proprio","possible_contact_after_any_close"):
            values=h.get(field,{})
            if set(values)!={"left","right"} or any(v is not False for v in values.values()):
                raise ValueError("Unknown or loaded replay state")
        plan=plan[index:]  # No completion claims or stale target/evidence are copied.
    else:
        if plan[0].kind!="pick" or last.part!=plan[0].hand or last.move!="close":
            raise ValueError("This diagnostic supports only an unverified first PICK after CLOSE")
        active=("left","right") if last.part=="both" else (last.part,)
        for arm in active:
            if actions[-1,14 if arm=="left" else 22]!=-1:raise ValueError("Saved close latch is not closed")
    return {"actions":actions,"plan":plan,"last_action":last,"expected_q":q,"expected_gripper":g,
            "receipt":{"spec_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_run":str(root),
                "source_sha256":spec["sha256"],"replay_controls":count,"replay_decisions":n,
                "autonomous_progress":False,"scene_truth_used":False,
                "purpose":"matched_unladen_approach_diagnostic" if approach else "matched_grasp_feedback_diagnostic"}}


def check_endpoint(replay,state):
    q=float(np.max(np.abs(state.q-replay["expected_q"])))
    g=float(np.max(np.abs(state.gripper-replay["expected_gripper"])))
    return {"q_max_abs_rad":q,"finger_max_abs_m":g,"q_limit_rad":.01,"finger_limit_m":.003,
            "passed":q<=.01 and g<=.003,"not_proof_of_same_scene_state_or_grasp":True}


def bootstrap_unverified_pick(manager,replay,state):
    if manager.index!=0 or manager.goal.kind!="pick":raise ValueError("Fresh first-pick harness required")
    if replay["receipt"]["purpose"]=="matched_unladen_approach_diagnostic":
        if np.any(state.gripper<.0495):raise ValueError("Replay is not currently unloaded/open")
        manager.last_gripper=state.gripper.copy()
        manager.events.append({"event":"SAVED_UNLADEN_APPROACH_PREFIX","not_autonomous_progress":True,
                               "fresh_target_observation_required":True})
        # Remain in fresh SEARCH: no old goal completion, target, visual evidence,
        # last-action feedback, or attachment claim can cross this boundary.
        return
    manager.last_action=replay["last_action"]
    manager.last_gripper=state.gripper.copy()
    manager.transition("VERIFY_GRASP")
    for arm in manager.arms:manager.pending_grasp[arm]=True
    manager.feedback=None  # No fabricated successful grasp or feedback.
    manager.events.append({"event":"SAVED_PREFIX_UNVERIFIED_CLOSE","not_verified_holding":True})

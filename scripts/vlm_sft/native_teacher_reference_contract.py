"""Reset-free loading and bounded source contract for full expert references."""
import json
from pathlib import Path
import numpy as np
from common import sha
from native_teacher_policy import validate_train_group
from native_teacher_reference_prepare import SCHEMA
from native_reference_profile import validate_profile


def load_reference(folder,release,code,executor,counts,*,validate_inactive=False):
    folder=Path(folder).resolve();manifest_path=folder.parent/"preparation.json"
    # CPU preflight is explicitly separate from execution: an inactive release
    # is accepted ONLY by this named inspection mode, never the runner default.
    permission=(release.get("authorize_reference_replay") is False and release.get("reviewer")=="") if validate_inactive else (
        release.get("authorize_reference_replay") is True and isinstance(release.get("reviewer"),str) and bool(release["reviewer"]))
    if (not permission or release.get("code_commit")!=code or
            release.get("executor_digest")!=executor or
            sha(manifest_path)!=release.get("preparation_manifest_sha256")):
        raise ValueError("New exact-code/reference authorization required before reset")
    manifest=json.loads(manifest_path.read_text())
    if manifest["schema"]!=SCHEMA or manifest["status"]!="PREPARED_NO_RESET_NO_OUTCOME_NO_SEED":
        raise ValueError("Unknown prepared reference schema")
    rows=[r for r in manifest["sources"] if folder.name==f"task_{r['task']}"]
    if len(rows)!=1:raise ValueError("Reference source ownership mismatch")
    row=rows[0]
    profile=validate_profile(release,manifest,row)
    if [row[k] for k in ("task","episode","instance")]!=release["source"]:
        raise ValueError("Authorization is for another instance/episode")
    expected={"prefix.npy","segment.npy","source_states.npy","teacher_reference.json","private_spec.json","segment_labels.json","label_selection_audit.json","window.json"}
    if set(row["files_sha256"])!=expected or any(sha(folder/k)!=v for k,v in row["files_sha256"].items()):
        raise ValueError("Prepared source bytes changed")
    ref=json.loads((folder/"teacher_reference.json").read_text());spec=json.loads((folder/"private_spec.json").read_text())
    validate_train_group(ref,counts,release["held_out_instance_groups"])
    source=ref["source"];skill=json.loads(ref["private_original_semantic_json"])
    if (len(skill)!=1 or skill[0].get("unbound_relation") or spec["verb"] not in ("GRASP","PRESS") or
            any(spec[k]!=skill[0][k] for k in ("verb","target","destination")) or
            any(source[k]!=row[k] for k in ("task","episode","instance")) or
            spec["hand"]!=row["hand"] or spec["support_hand"]!=row["support_hand"] or
            ref["source_label"]["segment_start"]!=row["start"] or ref["source_label"]["segment_end"]!=row["end"]):
        raise ValueError("Reference identity/intent changed")
    prefix,segment,states=(np.load(folder/(name+".npy"),allow_pickle=False) for name in ("prefix","segment","source_states"))
    for values,shape in ((prefix,(row["start"],23)),(segment,(row["end"]-row["start"],23)),(states,(len(segment)+1,61))):
        if values.shape!=shape or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError("Full reference arrays invalid")
    window=json.loads((folder/"window.json").read_text())
    if (window["official_mode"]!="train" or window["instance_id"]!=row["instance"] or window["seed"]!=0 or
            window["max_steps"]!=row["end"]+14 or window["prefix_actions_sha256"]!=row["files_sha256"]["prefix.npy"] or
            Path(window["prefix_actions_path"]).resolve()!=folder/"prefix.npy"):
        raise ValueError("Reference reset/window clock identity mismatch")
    budget=release["budget"]
    if profile is None and (budget["max_controls"]!=row["end"]+13 or budget["tail_controls"]!=12 or budget["final_hold_controls"]!=1 or
            not 1<=budget["seconds_after_reset"]<=2100 or not 1<=budget["run_MiB"]<=128 or
            not 1<=budget["total_MiB"]<=512 or release.get("model_calls")!=0 or release.get("resets")!=1):
        raise ValueError("Reference budget differs from the single registered full segment")
    gates=[json.loads(Path(p).read_text()) for p in release["engineering_gate_paths"]]
    if len(gates)!=2 or {g["task"] for g in gates}!={0,3} or any(g.get("gate_ok") is not True or
            g.get("robot_geometry_guards") is not True or g.get("implementation_digest")!=executor for g in gates):
        raise ValueError("Reviewed matching safety gates required")
    binding={"reference_preparation_sha256":sha(manifest_path),"source":row,"window":window}
    if profile is not None:binding.update(reference_profile=profile,purpose=manifest["purpose"])
    return ref,spec,prefix,segment,states,binding


def check_actual_joint_bounds(current,model):
    """Actual native joint safety, independent of exported demonstration lag.

    This is not collision/path certification and never clips a measured state.
    The 1e-5 rad allowance is floating-point boundary tolerance, not the old
    20 mrad source-reproduction criterion. Reference replay is not SafeServo BC.
    """
    q,g=np.asarray(current.q,float),np.asarray(current.gripper,float)
    lo,hi=np.asarray(model.lower,float),np.asarray(model.upper,float)
    if (q.shape!=(18,) or g.shape!=(2,) or lo.shape!=(18,) or hi.shape!=(18,) or
            not all(np.isfinite(x).all() for x in (q,g,lo,hi)) or np.any(lo>=hi)):
        raise ValueError("Finite calibrated actual reference state/bounds required")
    if np.any(q<lo-1e-5) or np.any(q>hi+1e-5):
        raise RuntimeError("ACTUAL_REFERENCE_JOINT_LIMIT_VIOLATION")


def source_state_diagnostic(current,source,frame_index):
    """Compare the FIXED source frame; never shift/search or earn success.

    LeRobot playback observations are not guaranteed to reproduce the dynamics
    of a fresh action replay. Source identity is checked by exact action bytes,
    instance/seed/interval and hashes; physical outcomes use actual measurements.
    """
    source=np.asarray(source,float)
    q,g=np.asarray(current.q,float),np.asarray(current.gripper,float)
    if (type(frame_index) is not int or frame_index<0 or source.shape!=(61,) or
            q.shape!=(18,) or g.shape!=(2,) or
            not all(np.isfinite(x).all() for x in (source,q,g))):
        raise ValueError("Finite fixed-frame source diagnostic required")
    dq=q-np.r_[source[53:57],source[3:10],source[28:35]]
    dg=g-np.array([source[24:26].mean(),source[49:51].mean()])
    return {"source_frame_index":frame_index,"q_error_rad":dq.tolist(),"gripper_error_m":dg.tolist(),
            "q_max_abs":float(abs(dq).max()),"gripper_max_abs":float(abs(dg).max()),
            "nearest_frame_search_used":False,"not_safety_or_outcome_gate":True}

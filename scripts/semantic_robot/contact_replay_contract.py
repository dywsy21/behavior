"""Strict frozen-action provenance for an audit-only simulator replay."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_contract(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    if spec.get("schema") != "contact_replay_audit_v1" or spec.get("autonomous_policy") is not False:
        raise ValueError("Explicit non-policy diagnostic required")
    n, count = spec["decisions"], spec["replay_controls"]
    if type(n) is not int or not 1 <= n <= 96 or type(count) is not int or not 1 <= count <= 1535:
        raise ValueError("Bounded replay required")
    root = Path(spec["source_run"])
    names = {"manifest.json", "result.json", "steps.jsonl", "sensor_checks.json"}
    names.update(f"decision_{i:03d}/proprio.json" for i in range(n))
    if set(spec["sha256"]) != names:
        raise ValueError("Exact manifest/result/trace/all boundary proprio identities required")
    data = {}
    for name in sorted(names):
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"][name]:
            raise ValueError("Source SHA mismatch: " + name)
        data[name] = raw.decode()
    manifest, result = (json.loads(data[k]) for k in ("manifest.json", "result.json"))
    if (manifest["code_commit"] != spec["source_code_commit"] or manifest["task"] != 0 or
            manifest["split"] != "train" or manifest["seed"] != 0 or manifest["instance"] != 138 or
            manifest["args"]["prefix"] != 0 or manifest["args"].get("replay_prefix_spec") is not None or
            manifest["args"].get("odometry_substep_controls") != 6 or
            result["prefix_controls"] != 0 or result["diagnostic_replay_controls"] != 0):
        raise ValueError("Exact original-start development source required")
    rows = [json.loads(line) for line in data["steps.jsonl"].splitlines()]
    controls = [r for r in rows if "action23" in r]
    commands = [r for r in rows if "action" in r and "feedback" in r]
    if (len(controls) != count or [r["control"] for r in controls] != list(range(1, count + 1)) or
            len(commands) != n or [r["decision"] for r in commands] != list(range(n)) or
            commands != result["decisions"] or any(r.get("terminal") for r in controls)):
        raise ValueError("Missing, duplicate, reordered or terminated source trace")
    if result["controls"] != count + 1 or rows[-1] != {"control": count + 1, "safety_stop": True}:
        raise ValueError("Expected exactly one separately recorded original final hold")
    actions = np.asarray([r["action23"] for r in controls], dtype=float)
    if actions.shape != (count, 23) or not np.isfinite(actions).all() or np.any(np.abs(actions[:, [0, 1, 2, 14, 22]]) > 1 + 1e-6):
        raise ValueError("Finite native23 commands and normalized base/grip required")
    boundaries = {}
    sample_labels = ["after_prefix", "decision_0"]
    sample_controls, end_controls = set(), set()
    end = 0
    for i, row in enumerate(commands):
        selected = [r for r in controls if r["decision"] == i]
        if (not selected or row["control_start"] != end or row["control_end"] != selected[-1]["control"] or
                [r["control"] for r in selected] != list(range(end + 1, row["control_end"] + 1))):
            raise ValueError("Action clock does not close")
        end = row["control_end"]
        end_controls.add(end)
        ticks = list(range(row["control_start"] + 6, end + 1, 6))
        if not ticks or ticks[-1] != end:
            ticks.append(end)
        sample_controls.update(ticks)
        sample_labels.extend(f"motion_{i}_{tick}" for tick in ticks)
        saved = json.loads(data[f"decision_{i:03d}/proprio.json"])
        q, grip = np.asarray(saved["q"], dtype=float), np.asarray(saved["gripper"], dtype=float)
        if q.shape != (18,) or grip.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(grip).all():
            raise ValueError("Finite boundary proprio required")
        boundaries[row["control_start"]] = {"decision": i, "q": q, "gripper": grip, "action": row["action"]}
    if end != count:
        raise ValueError("Final control uncovered")
    sensors = json.loads(data["sensor_checks.json"])
    if [r["label"] for r in sensors] != sample_labels:
        raise ValueError("Source observation clock differs from fixed six-tick replay")
    if any(r["cameras"][v]["render_barrier_updates"] != 4 for r in sensors for v in ("head", "left_wrist", "right_wrist")):
        raise ValueError("Source render-only barrier differs")
    return {"spec": spec, "manifest": manifest, "actions": actions, "boundaries": boundaries,
            "sample_controls": sample_controls, "end_controls": end_controls,
            "commands": commands, "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest()}

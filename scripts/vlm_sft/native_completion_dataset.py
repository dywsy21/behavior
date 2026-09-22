"""H09AA:39 offline completion sidecars from exactly five reviewed TRAIN runs.

No collection, model, simulator, training or runtime early-stop behavior.
Private outcomes stay in offline provenance, never actor/text/images.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import numpy as np

from common import CAMERAS, sha, write_json
import native_actor_protocol as protocol
from native_completion import VERSION, SIDECAR_VERSION, loss_mask, validate_sidecar
from native_dataset import load_dataset, verified_run, checked_images, read, lines
from native_motion_codec import ACTOR_VERSION, VERSION as MOTION_CODEC
from native_teacher_artifacts import load_calibration
from native_teacher_group_prepare import COUNTS_SHA
from native_teacher_outcomes import LocalOutcome

DATASET_SHA = "5b2b5ee01f1f325cc5f35b1665a014170237f284c605038d75ad2a40ad0d693c"
ROWS_SHA = "ef1a62fad44c8bab863a6f6ec7548aab4817f9b1c806e226d0ab81dfef1310cc"
SOURCES_SHA = "5e5a91b8c866708a0f62af33c59551263b8512dd62a08075fee4726604abd65b"
EXPECTED = ((1, 192, 396, 10), (1, 114, 989, 4), (1, 114, 969, 6),
            (1, 192, 388, 6), (1, 114, 953, 8))


def readable_tree(root):
    """Do not inherit rglob's silent unreadable-directory behavior."""
    pending = [Path(root)]
    while pending:
        path = pending.pop(); mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not mode & 0o444 or not os.access(path, os.R_OK):
            raise ValueError("Symlink or unreadable source evidence")
        if stat.S_ISDIR(mode):
            if not mode & 0o111 or not os.access(path, os.X_OK):
                raise ValueError("Untraversable source evidence")
            with os.scandir(path) as entries:
                pending.extend(Path(e.path) for e in entries)
        elif not stat.S_ISREG(mode):
            raise ValueError("Only regular evidence files allowed")


def state_label(verdict, motion):
    if (type(verdict.get("tick")) is not int or verdict["tick"] < 0 or
            type(verdict.get("stable_ticks")) is not int or verdict["stable_ticks"] < 0 or
            type(verdict.get("required_ticks")) is not int or verdict["required_ticks"] < 12):
        raise ValueError("Known same-physics-clock outcome/stability evidence required")
    status = verdict.get("outcome")
    if status == "IN_PROGRESS" and motion is not None:
        return {"skill_status": "CONTINUE", "motion": motion}
    if (status == "SUCCEEDED" and motion is None and type(verdict.get("stable_ticks")) is int
            and type(verdict.get("required_ticks")) is int and
            verdict["stable_ticks"] >= verdict["required_ticks"] >= 12):
        return {"skill_status": "REQUEST_VERIFY", "motion": None}
    raise ValueError("Unknown/failed/ambiguous outcome cannot become a completion label")


def terminal_hold_check(capture, hold, final, last_command):
    protocol.check_clock(capture["clock"])
    q = protocol.finite(capture["q"], (18,))
    command = protocol.finite(hold["action23"], (23,))
    previous = protocol.finite(last_command, (23,))
    if (hold.get("completed") is not True or type(hold.get("prefix_controls")) is not int or
            hold["prefix_controls"] != capture["clock"]["prefix_control"] or
            type(hold.get("native_controls")) is not int or hold["native_controls"] != capture["clock"]["native_control"] + 1 or
            type(final["frame"]["tick"]) is not int or final["frame"]["tick"] != sum(capture["clock"].values()) + 1 or
            not np.array_equal(np.r_[command[3:14], command[15:22]], q) or
            not np.array_equal(command[[14, 22]], previous[[14, 22]])):
        raise ValueError("Exact next one-control pose/grip-preserving hold required")
    state_label(final["verdict"], None)


def make_sidecar(row, label, evidence, *, terminal=False):
    value = {"schema": SIDECAR_VERSION, "completion_protocol": VERSION,
             "id": row["id"] + ("_terminal" if terminal else "_continue"),
             **{k: copy.deepcopy(row[k]) for k in ("protocol", "actor", "text", "images", "provenance")},
             "label": label, "loss_mask": loss_mask(label),
             "source_motion_row_id": None if terminal else row["id"],
             "offline_label_evidence": evidence}
    validate_sidecar(value); checked_images(value)
    return value


def completion_rows(entry, original):
    """Called only after whole-run replay exactly matches original motion rows."""
    run = Path(entry["run"]); result = read(run / "result.json")
    initial = read(run / "PRIVATE_teacher_initial.json")
    trace = lines(run / "PRIVATE_teacher_trace.jsonl")
    oracle = LocalOutcome(initial["spec"])
    states = {initial["frame"]["tick"]: oracle.update(initial["frame"])}
    states.update({r["frame"]["tick"]: r["verdict"] for r in trace})
    model = load_calibration(run); output = []; review_images = []
    paths = {name: {"path": str(run / name), "sha256": sha(run / name)} for name in
             ("manifest.json", "result.json", "PRIVATE_teacher_initial.json", "PRIVATE_teacher_trace.jsonl",
              "final_hold.json", "PRIVATE_final_hold_outcome.json", "native_trace.jsonl", "issued_trace.jsonl")}
    source = {"run": str(run), "inventory": entry["inventory"], "inventory_sha256": sha(entry["inventory"]),
              "parent_review": entry["review"], "parent_review_sha256": entry["review_sha256"],
              "files": paths, "not_actor_input": True}
    history = []
    for i, row in enumerate(original):
        folder = run / f"teacher_{i:02d}"
        before = read(folder / "before/capture.json"); tick = sum(before["clock"].values())
        if row["actor"]["history"] != history[-5:] or row["provenance"]["clock"] != before["clock"]:
            raise ValueError("Actually completed pre-macro history/clock differs")
        verdict = states.get(tick, {})
        value = make_sidecar(row, state_label(verdict, row["target"]),
                             {**source, "same_state_tick": tick, "verdict": verdict,
                              "capture_path": str(folder / "before/capture.json"),
                              "capture_sha256": sha(folder / "before/capture.json"),
                              "native_execution_sha256": sha(folder / "native_execution.json")})
        output.append(value); history.append(row["target"])
    if len(original) != len(result["samples"]) or not original:
        raise ValueError("Exactly the full successful sequence required")
    folder = run / f"teacher_{len(original)-1:02d}"
    capture_path = folder / "after_settle/capture.json"; capture = read(capture_path)
    tick = sum(capture["clock"].values()); verdict = states.get(tick, {})
    record = read(folder / "QUARANTINED_record.json")
    if verdict != record["local_outcome"] or tick != len(lines(run / "native_trace.jsonl")):
        raise ValueError("Terminal capture is not the verified final successful state")
    final = read(run / "PRIVATE_final_hold_outcome.json")
    terminal_hold_check(capture, read(run / "final_hold.json"), final,
                        lines(run / "native_trace.jsonl")[-1]["action23"])
    proprio, hashes, binding = protocol.from_capture(capture_path.parent, model,
        capture_sha256=sha(capture_path), expected_clock=capture["clock"], calibration_sha256=model.sha,
        protocol=ACTOR_VERSION)
    base = original[-1]; actor = protocol.actor_input(base["actor"]["task"], base["actor"]["active_instruction"],
        proprio, hashes, history[-5:], protocol=ACTOR_VERSION)
    row = {**protocol.inference_row(actor), "id": base["id"],
           "images": {v: str(capture_path.parent / (v + ".png")) for v in CAMERAS},
           "provenance": {**base["provenance"], **binding}}
    terminal = make_sidecar(row, state_label(verdict, None),
        {**source, "same_state_tick": tick, "verdict": verdict, "capture_path": str(capture_path),
         "capture_sha256": sha(capture_path), "following_actual_hold_controls": 1,
         "hold_is_not_a_motion_label": True}, terminal=True)
    if any(r["provenance"]["capture_sha256"] == binding["capture_sha256"] for r in original):
        raise ValueError("Terminal state duplicates an original motion input")
    output.append(terminal)
    for item in (output[-2], terminal):
        review_images.append({"id": item["id"], "label": item["label"], "clock": item["provenance"]["clock"],
            "history": item["actor"]["history"], "capture": item["offline_label_evidence"]["capture_path"],
            "capture_sha256": item["provenance"]["capture_sha256"], "run_inventory_sha256": source["inventory_sha256"],
            "parent_review_sha256": source["parent_review_sha256"],
            "images": {v: {"path": item["images"][v], "sha256": item["actor"]["current_rgb_sha256"][v]} for v in CAMERAS}})
    return output, review_images, source


def build(dataset, sources, counts, output):
    started = time.monotonic(); dataset, sources, counts, output = map(Path, (dataset, sources, counts, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (sha(dataset / "dataset.json") != DATASET_SHA or sha(dataset / "train.jsonl") != ROWS_SHA or
            sha(sources) != SOURCES_SHA or sha(counts) != COUNTS_SHA):
        raise ValueError("Only the frozen five-run34-row TRAIN dataset is authorized")
    original, manifest = load_dataset(dataset, require_gate=True)
    entries = read(sources)
    if len(entries) != 5 or len(original) != 34 or manifest["source_list_sha256"] != SOURCES_SHA:
        raise ValueError("Exactly five original trajectories and34 motions required")
    expected = tuple((*r["group"], r["prefix"], r["macros"]) for r in manifest["runs"])
    if expected != EXPECTED:
        raise ValueError("Unregistered TRAIN start or heldout source")
    sidecars = []; images = []; bindings = []
    for entry, registered in zip(entries, manifest["runs"]):
        readable_tree(entry["run"])
        if output.resolve().is_relative_to(Path(entry["run"]).resolve()):
            raise ValueError("Never write completion data inside old evidence")
        rows, receipt = verified_run(entry, read(counts), export_codec=MOTION_CODEC)
        batch = [r for r in original if r["provenance"]["run_inventory_sha256"] == registered["inventory_sha256"]]
        if rows != batch or receipt != registered:
            raise ValueError("Replayed whole-run/motion rows differ from frozen original data")
        new, views, binding = completion_rows(entry, batch)
        sidecars.extend(new); images.extend(views); bindings.append(binding)
    if (len(sidecars) != 39 or len({r["id"] for r in sidecars}) != 39 or len(images) != 10 or
            sum(r["label"]["skill_status"] == "CONTINUE" for r in sidecars) != 34):
        raise ValueError("Incomplete or duplicate completion sidecars")
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Original dataset is immutable")
    payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in sidecars)
    if len(payload.encode()) > 8 * 1024**2:
        raise ValueError("Bounded small sidecar output exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "states.jsonl").write_text(payload)
    write_json(output / "manual_review.json", {"schema": SIDECAR_VERSION, "states": images,
        "three_views_per_state": True, "parent_manual_review_pending": True})
    source_root = Path(__file__).resolve().parents[2]
    result = {"schema": SIDECAR_VERSION, "completion_protocol": VERSION, "actor_protocol": ACTOR_VERSION,
        "motion_codec": MOTION_CODEC, "status": "CANDIDATE_PENDING_PARENT_REVIEW_NO_TRAINING",
        "source_commit": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "original_dataset_sha256": DATASET_SHA, "original_rows_sha256": ROWS_SHA,
        "original_sources_sha256": SOURCES_SHA, "counts_sha256": COUNTS_SHA,
        "states_sha256": sha(output / "states.jsonl"), "manual_review_sha256": sha(output / "manual_review.json"),
        "rows": 39, "CONTINUE": 34, "REQUEST_VERIFY": 5, "motion_loss_rows": 34,
        "original_motion_supervision_unchanged": True, "original_coverage": manifest["coverage"],
        "sources": bindings, "completion_dataset_released": False,
        "request_verify_is_success": False, "runtime_early_stop_implemented": False,
        "new_physical_controls": 0, "new_model_calls": 0, "new_training_updates": 0,
        "actual_python": sys.executable, "seconds": time.monotonic() - started}
    write_json(output / "completion_dataset.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("dataset", "sources", "counts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.sources, args.counts, args.output), indent=2))

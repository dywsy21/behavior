"""Prepare exactly three TRAIN pilot sources, without simulator/model imports.

Use source metadata only; output is teacher reference material, not SFT data.
Prefix conversion and small source clips are explicitly counted artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from common import CAMERAS, sha, skill_text, write_json
from native_teacher_contract import SCHEMA, select_sources, compile_candidates
from prepare import ROOT, LABELS, RELEASE


def prepare(config, counts, output, robot_template, previews=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    template = json.loads(Path(robot_template).read_text())
    selected = select_sources(counts)
    meta = [r for p in sorted((ROOT / "meta/episodes").rglob("*.parquet")) for r in pq.read_table(p).to_pylist()]
    quarantine = pq.read_table(RELEASE / "quarantine_ranges.parquet").to_pylist()
    if sha(LABELS) != counts["source_identity"]["labels_sha256"] or sha(RELEASE / "quarantine_ranges.parquet") != counts["source_identity"]["quarantine_sha256"]:
        raise ValueError("Reviewed TRAIN annotation identity changed")
    prepared = []
    for pilot, source in zip(config["pilots"], selected):
        if any(pilot[k] != source[k] for k in ("task", "episode", "instance")):
            raise ValueError("Preselected source changed")
        e = next(e for e in meta if e["episode_index"] == pilot["episode"])
        if e["task_instance_id"] != source["instance"] or e["task_index"] != source["task"]:
            raise ValueError("Episode/instance mismatch")
        table = pq.read_table(source["parquet"], filters=[("episode_index", "=", pilot["episode"])],
                              columns=["frame_index", "action", "observation.state"]).sort_by("frame_index")
        s, a = (np.asarray(table[k].to_pylist(), dtype=float) for k in ("observation.state", "action"))
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(e["length"])):
            raise ValueError("Observation/action frame alignment")
        columns = ["frame_index", "active_skills_semantic_json", "low_action_supervision_mask",
                   "action_horizon_end", "segment_start", "segment_end", "memlite_branch", "source_kind"]
        labels = pq.read_table(LABELS, filters=[("episode_index", "=", pilot["episode"])], columns=columns).to_pylist()
        h = hashlib.sha256(s.tobytes()+a.tobytes())
        h.update(json.dumps(labels, sort_keys=True, separators=(",", ":")).encode())
        if h.hexdigest() != source["extracted_arrays_and_labels_sha256"]:
            raise ValueError("Exact H09R TRAIN arrays/annotations changed")
        f = pilot["frame"]
        by_frame = {r["frame_index"]: r for r in labels}
        label = by_frame[f]
        if label["segment_start"] != f or not any(x["verb"] == pilot["verb"] for x in json.loads(label["active_skills_semantic_json"])):
            raise ValueError("Registered phase start changed")
        for index in range(f, f+17):
            r = by_frame[index]
            if (r["source_kind"] != "original_demo" or r["memlite_branch"] != "low" or
                    not r["low_action_supervision_mask"] or r["active_skills_semantic_json"] != label["active_skills_semantic_json"] or
                    min(r["segment_end"], r["action_horizon_end"]) <= f+16):
                raise ValueError("Same-skill released source required for all reference frames")
        if any(r["episode_index"] == pilot["episode"] and r["frame_start"] <= f+16 and r["frame_end"] >= f for r in quarantine):
            raise ValueError("Quarantined source")
        folder = output / f"task_{pilot['task']}"
        folder.mkdir()
        np.save(folder / "prefix.npy", a[:f].astype(np.float32), allow_pickle=False)
        reference = {"schema": SCHEMA, "source": source, "pilot": pilot,
                     "active_instruction": skill_text(label["active_skills_semantic_json"]),
                     "private_original_semantic_json": label["active_skills_semantic_json"],
                     "source_label": label, "source_states": s[f:f+17].tolist(),
                     "source_actions": a[f:f+16].tolist(), "prefix_controls": f,
                     "proposals": compile_candidates(s[f:f+17], a[f:f+16]),
                     "source_review_media": {}, "training_eligible": False}
        for view, camera in CAMERAS.items():
            key = "observation.rgb."+camera
            stem = "videos/"+key
            movie = ROOT / f"videos/{key}/chunk-{e[stem+'/chunk_index']:03d}/file-{e[stem+'/file_index']:03d}.mp4"
            start = e[stem+"/from_timestamp"] + f/30
            reference["source_review_media"][view] = {"video": str(movie), "start_seconds": start,
                                                     "frame_start": f, "frame_end": f+16}
            if previews:
                target = folder / ("expert_"+view+".mp4")
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(movie),
                                "-frames:v", "17", "-an", "-vf", "scale=384:-2", "-c:v", "libx264", "-threads", "1", str(target)], check=True, timeout=60)
                reference["source_review_media"][view].update(preview=str(target), sha256=sha(target))
        write_json(folder / "teacher_reference.json", reference)
        window = {k: v for k, v in template.items()}
        window.update(window_id=f"h09s-train-t{pilot['task']}-i{pilot['instance']}-f{f}",
                      task_name=e["tasks"][0], official_mode="train", instance_id=pilot["instance"], seed=0,
                      max_steps=f+201, max_chunks=1, execute_steps=1,
                      prefix_actions_path=str((folder / "prefix.npy").resolve()), prefix_actions_sha256=sha(folder / "prefix.npy"),
                      semantic_subgoal={"parent_goal": e["tasks"][0], "active_skills_semantic_json": "[]",
                                        "active_skills_text": reference["active_instruction"]})
        write_json(folder / "window.json", window)
        prepared.append({**pilot, "reference_sha256": sha(folder / "teacher_reference.json"),
                         "window_sha256": sha(folder / "window.json"), "prefix_sha256": sha(folder / "prefix.npy"),
                         "ranked_proposals": reference["proposals"]["ranked_proposals"],
                         "proposal_reason": reference["proposals"]["reason"]})
    result = {"schema": SCHEMA, "status": "PREPARED_NOT_COLLECTED_NOT_TRAINING_DATA", "sources": prepared,
              "new_models": 0, "new_controls": 0, "new_resets": 0,
              "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
    write_json(output / "preparation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--counts", required=True)
    p.add_argument("--output", required=True); p.add_argument("--robot-template", required=True)
    p.add_argument("--previews", action="store_true")
    x = p.parse_args()
    print(json.dumps(prepare(json.loads(Path(x.config).read_text()), json.loads(Path(x.counts).read_text()),
                             x.output, x.robot_template, x.previews), ensure_ascii=False))

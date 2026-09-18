"""Bounded expert index and image extraction; no model rollouts become labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq

from common import CAMERAS, VERSION, TOKENS, actor_state, classify_window, prompt, sha, skill_text, write_json

ROOT = Path("/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4")
LABELS = Path("/mnt/sdc1/robodojo/datasets/memlite_skill_annotations_task0_4_v6_clean_r2_20260909/meta/memlite_skill_annotations_v6.parquet")
RELEASE = Path("/mnt/sdc1/robodojo/datasets/memlite_skill_annotations_task0_4_v6_formal_a_composite_release_v2_20260909/meta")
REPO = Path(__file__).resolve().parents[2]


def reserve(path):
    if shutil.disk_usage(path).free < 80*1024**3:
        raise RuntimeError("Disk reserve below 80 GiB")


def choose_groups(episodes, cfg):
    excluded = {tuple(v) for v in cfg["exclude_development_instances"]}
    groups = {}
    for task in cfg["tasks"]:
        rows = sorted([e for e in episodes if e["task_index"] == task], key=lambda e: e["episode_index"])
        if len(rows) != 200:
            raise ValueError("Original task group no longer contains 200 episodes")
        original_eval = {r["episode_index"] for r in rows[-10:]}
        original_eval_instances = {r["task_instance_id"] for r in rows[-10:]}
        candidates = [r for r in rows[:-10] if (task, r["task_instance_id"]) not in excluded
                      and r["task_instance_id"] not in original_eval_instances]
        # Hash grouping is independent of action/visual labels and model scores.
        candidates.sort(key=lambda r: hashlib.sha256(f'{cfg["seed"]}:{task}:{r["task_instance_id"]}'.encode()).hexdigest())
        seen = set()
        unique = []
        for r in candidates:
            if r["task_instance_id"] not in seen:
                unique.append(r)
                seen.add(r["task_instance_id"])
        n = 0
        for split, count in (("train", cfg["train_episodes_per_task"]), ("validation", cfg["validation_episodes_per_task"]), ("test", cfg["test_episodes_per_task"])):
            for r in unique[n:n+count]:
                groups[r["episode_index"]] = {"split": split, "episode": r}
            n += count
        if len(unique) < n or any(k in original_eval for k in groups):
            raise ValueError("Split selection failed")
    instance_splits = defaultdict(set)
    for v in groups.values():
        e = v["episode"]
        instance_splits[(e["task_index"], e["task_instance_id"])].add(v["split"])
    if any(len(x) != 1 for x in instance_splits.values()):
        raise ValueError("Source-instance leakage")
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--reuse-images-from",type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["minimum_sample_spacing_frames"] < cfg["horizon_frames"]:
        raise ValueError("Stride must be at least horizon; overlapping future labels cannot become history")
    if args.extract:
        return extract(args.output,args.reuse_images_from)
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use clean pinned source")
    args.output.mkdir(parents=True, exist_ok=False)
    reserve(args.output)
    started = time.time()
    episodes = [r for f in sorted((ROOT/"meta/episodes").rglob("*.parquet")) for r in pq.read_table(f).to_pylist()]
    groups = choose_groups(episodes, cfg)
    write_json(args.output/"groups.json", {str(k):v for k,v in groups.items()})
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, (ROOT/"meta/tasks.jsonl").read_text().splitlines())}
    # The released quarantine is never promoted into a positive teacher.
    quarantine = pq.read_table(RELEASE/"quarantine_ranges.parquet").to_pylist()
    write_json(args.output/"quarantine_schema.json", {"schema": str(pq.read_schema(RELEASE/"quarantine_ranges.parquet")), "first_rows":quarantine[:2]})
    invalid_ranges = defaultdict(list)
    for r in quarantine:
        ep = r.get("episode_index")
        start = r.get("start_frame", r.get("frame_start", r.get("start")))
        end = r.get("end_frame", r.get("frame_end", r.get("end")))
        if ep is None or start is None or end is None:
            raise ValueError("Unknown released quarantine range schema: " + str(r))
        invalid_ranges[int(ep)].append((int(start), int(end)))
    h = cfg["horizon_frames"]
    samples, rejects, hist = [], Counter(), Counter()
    rejects_by_task=defaultdict(Counter);total_by_task=Counter();accepted_by_task=Counter()
    for number, (ep, group) in enumerate(sorted(groups.items()), 1):
        reserve(args.output)
        e = group["episode"]
        file = ROOT/f'data/chunk-{e["data/chunk_index"]:03d}/file-{e["data/file_index"]:03d}.parquet'
        data = pq.read_table(file, filters=[("episode_index", "=", ep)], columns=["frame_index", "action", "observation.state"]).sort_by("frame_index")
        states = np.asarray(data["observation.state"].to_pylist())
        actions = np.asarray(data["action"].to_pylist())
        frames = data["frame_index"].to_numpy()
        if not np.array_equal(frames, np.arange(e["length"])):
            raise ValueError("Episode frame alignment changed")
        columns = ["frame_index", "active_skills_semantic_json", "low_action_supervision_mask", "action_horizon_end", "segment_end", "memlite_branch"]
        labels = {r["frame_index"]: r for r in pq.read_table(LABELS, filters=[("episode_index", "=", ep)], columns=columns).to_pylist()}
        previous = []
        candidates = []
        for f in range(0, len(states)-h, cfg["minimum_sample_spacing_frames"]):
            total_by_task[e["task_index"]]+=1
            label = labels.get(f)
            if (not label or not label["low_action_supervision_mask"] or label["memlite_branch"] != "low"
                    or min(label["action_horizon_end"], label["segment_end"]) <= f+h
                    or any(lo <= f+h and hi >= f for lo,hi in invalid_ranges[ep])):
                rejects["not_released_same_skill_window"] += 1
                rejects_by_task[e["task_index"]]["not_released_same_skill_window"]+=1
                previous = []
                continue
            try:
                active = skill_text(label["active_skills_semantic_json"])
            except ValueError:
                rejects["unbound_instruction"] += 1
                rejects_by_task[e["task_index"]]["unbound_instruction"]+=1
                previous = []
                continue
            token, evidence = classify_window(states[f:f+h+1], actions[f:f+h], actions[f-1] if f else None)
            if token is not None and token not in TOKENS:
                token,evidence=None,dict(evidence,reject="executor_contract_mismatch")
            if token is None:
                rejects[evidence["reject"]] += 1
                rejects_by_task[e["task_index"]][evidence["reject"]]+=1
                previous = []  # do not invent an executed semantic history for mixed actions
                continue
            state = actor_state(states[f])
            row = {"id": f"t{e['task_index']}_e{ep}_f{f}", "split": group["split"], "task_id": e["task_index"],
                   "episode_index": ep, "instance_id": e["task_instance_id"], "frame_index": f,
                   "future_frame": f+h, "source": "original_expert", "task":tasks[e["task_index"]],
                   "active_instruction":active, "proprio":state, "history":previous[-5:], "target":token,
                   "text":prompt(tasks[e["task_index"]], active, state, previous[-5:]),
                   "label_evidence": evidence, "same_skill_end_exclusive":min(label["action_horizon_end"],label["segment_end"]),
                   "source_fingerprint":hashlib.sha256(states[f:f+h+1].tobytes()+actions[f:f+h].tobytes()+label["active_skills_semantic_json"].encode()).hexdigest()}
            candidates.append(row)
            accepted_by_task[e["task_index"]]+=1
            previous.append(token)
        # Class round robin within source episode prevents stationary/base-heavy
        # clips from filling the bounded dataset. Evaluation reports macro and
        # per-class counts, not a population success estimate.
        buckets = defaultdict(list)
        for row in candidates:
            buckets[row["target"]].append(row)
        rng = random.Random(cfg["seed"]+ep)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        chosen=[]
        while buckets and len(chosen) < cfg["max_samples_per_episode"]:
            for key in sorted(list(buckets)):
                if len(chosen) >= cfg["max_samples_per_episode"]:
                    break
                chosen.append(buckets[key].pop())
                if not buckets[key]:
                    del buckets[key]
        samples.extend(sorted(chosen, key=lambda r:r["frame_index"]))
        hist.update((r["split"], r["target"]) for r in chosen)
        print(json.dumps({"episodes_complete":number,"episode":ep,"split":group["split"],"candidates":len(candidates),"kept":len(chosen)}),flush=True)
    for split in ("train", "validation", "test"):
        rows = [r for r in samples if r["split"] == split]
        with (args.output/(split+".jsonl")).open("x") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    manifest={"version":VERSION,"status":"indexed_not_image_reviewed","code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
              "config":cfg,"config_sha256":sha(args.config),"labels_path":str(LABELS),"release_manifest_sha256":sha(RELEASE/"composite_release_manifest.json"),
              "quarantine_sha256":sha(RELEASE/"quarantine_ranges.parquet"),"root":str(ROOT),"group_sha256":sha(args.output/"groups.json"),
              "counts":dict(Counter(r["split"] for r in samples)),"class_counts":{s:dict(Counter(r["target"] for r in samples if r["split"]==s)) for s in ("train","validation","test")},
              "rejected_windows":dict(rejects),"elapsed_s":time.time()-started,
              "coverage_by_task":{str(t):{"all_windows":total_by_task[t],"accepted_before_cap":accepted_by_task[t],"rejected":dict(rejects_by_task[t])} for t in cfg["tasks"]},
              "split_sha256":{s:sha(args.output/(s+".jsonl")) for s in ("train","validation","test")},
              "limitations":["Conservative projection of continuous expert direction; not exact primitive demonstrations.","No positive terminal or outcome labels.","Instance-group holdout is within three known tasks, not task-level generalization."]}
    write_json(args.output/"index_manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def extract(out,reuse=None):
    import av
    from PIL import Image
    manifest=json.loads((out/"index_manifest.json").read_text())
    groups=json.loads((out/"groups.json").read_text())
    media=out/"images"
    media.mkdir(exist_ok=False)
    counts=Counter()
    rows_by_ep=defaultdict(list)
    cached={}
    if reuse:
        for split in ("train","validation","test"):
            for line in (reuse/(split+"_images.jsonl")).read_text().splitlines():
                row=json.loads(line);cached[row["id"]]=row
    all_rows={}
    for split in ("train","validation","test"):
        rows=[json.loads(x) for x in (out/(split+".jsonl")).read_text().splitlines()]
        all_rows[split]=rows
        for row in rows:
            rows_by_ep[row["episode_index"]].append(row)
    for ep,rows in sorted(rows_by_ep.items()):
        reserve(out)
        e=groups[str(ep)]["episode"]
        for view,camera in CAMERAS.items():
            key="videos/observation.rgb."+camera
            path=ROOT/f'{key}/chunk-{e[key+"/chunk_index"]:03d}/file-{e[key+"/file_index"]:03d}.mp4'
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                stream.thread_type="AUTO"
                stream.codec_context.thread_count=2
                for row in rows:
                    old=cached.get(row["id"])
                    if old is not None:
                        source=reuse/old["images"][view]
                        if sha(source)!=old["image_receipts"][view]["png_sha256"]:raise RuntimeError("Reused immutable frame changed")
                        row.setdefault("images",{})[view]=str(source.resolve())
                        row.setdefault("image_receipts",{})[view]=old["image_receipts"][view]
                        counts[view]+=1
                        continue
                    timestamp=float(e[key+"/from_timestamp"])+row["frame_index"]/30
                    container.seek(int(timestamp/stream.time_base),stream=stream,backward=True)
                    found=None
                    for frame in container.decode(stream):
                        actual=float(frame.pts*stream.time_base)
                        if actual >= timestamp-1/60:
                            found=(frame,actual)
                            break
                    if found is None or abs(found[1]-timestamp)>1/60+1e-4:
                        raise RuntimeError(f"Video/frame alignment failed: {path}:{timestamp}:{found}")
                    image=found[0].to_image().convert("RGB")
                    # Preserve square geometry. Model sees three 256px views,
                    # not a montage that changes positional coordinates.
                    image=image.resize((256,256),Image.Resampling.LANCZOS)
                    rel=f'images/{row["id"]}_{view}.png'
                    image.save(out/rel)
                    row.setdefault("images",{})[view]=rel
                    row.setdefault("image_receipts",{})[view]={"video":str(path),"requested_timestamp_s":timestamp,
                        "actual_timestamp_s":found[1],"source_frame_pts":found[0].pts,"png_sha256":sha(out/rel)}
                    counts[view]+=1
        print(json.dumps({"extracted_episode":ep,"samples":len(rows),"images":dict(counts)}),flush=True)
    for split,rows in all_rows.items():
        with (out/(split+"_images.jsonl")).open("x") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    write_json(out/"image_manifest.json",{"status":"extracted_pending_manual_review","index_manifest_sha256":sha(out/"index_manifest.json"),
        "image_counts":dict(counts),"splits":{s:sha(out/(s+"_images.jsonl")) for s in all_rows},"all_frames_within_half_frame_tolerance":True})


if __name__ == "__main__":
    main()

"""CPU-only real all-task loader/token checks and locatable visual-review sheets."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import heapq
import json
import multiprocessing
import os
from pathlib import Path
import textwrap
import time
import numpy as np

from g05.data.memlite_stage1_dataset import Stage1Dataset
from g05.data.memlite_stage1_labels import anchor_records, projection
from g05.utils.training.stage1_model import configuration
from g05.utils.training.stage1_runtime import sha256, atomic_json

DATASETS = {}


def initialize(release, root):
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((Path(release)/"manifest.json").read_text())
    for branch in ("high", "low"):
        config = configuration(root, branch, manifest["task_names"])
        for split in ("train", "eval"):
            DATASETS[(branch, split)] = Stage1Dataset(release, config, branch, split)


def read_job(job):
    import torch
    from PIL import Image, ImageDraw, ImageFont
    split, index, output, ordinal = job
    record = {}
    for branch in ("high", "low"):
        dataset = DATASETS[(branch, split)]
        sample = dataset[index]
        record[branch] = sample["samples"]
        if branch == "low":
            outer = sample
            raw, identity = dataset.raw(index)
    for branch, sample in record.items():
        from g05.data_processor.processor.memlite_v6_projection import validate_embedded_model_projection
        validate_embedded_model_projection(sample)
    for key in ("parent_goal", "active_skills_semantic_json"):
        if record["high"][key] != record["low"][key]:
            raise ValueError("High target and low condition differ at the same raw state")
    if any(v.shape != (1, 3, 256, 256) or not torch.isfinite(v).all() for v in outer["pixel_values"].values()):
        raise ValueError("Invalid real RGB tensor contract")
    if tuple(torch.nonzero(outer["action_dim_is_pad"]).flatten().tolist()) != (7, 8, 17, 18):
        raise ValueError("Real control channel is masked")
    # Raw pixels are never changed. The following contact sheet is a separate
    # diagnostic artifact with identifiers and the exact model-visible label.
    sheet = Image.new("RGB", (960, 500), "white")
    for col, image in enumerate(raw["images"].values()):
        arr = image[0].permute(1, 2, 0).numpy()
        sheet.paste(Image.fromarray(arr).resize((320, 320)), (col * 320, 0))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    title = f"{ordinal} {split} task={identity['task']} ep={identity['episode']} frame={identity['frame']} valid_actions={identity['valid_action_steps']}"
    lines = [title, raw["task"], *textwrap.wrap(raw["model_projection"]["active_skills_text"], 115),
             *textwrap.wrap(raw["model_projection"]["parent_goal"], 115)]
    for line_number, line in enumerate(lines[:10]):
        draw.text((8, 324 + line_number * 17), line, fill="black", font=font)
    path = Path(output) / "images" / f"{ordinal:04d}-{split}-{index}.jpg"
    sheet.save(path, quality=90)
    # A compact real model sample is returned for CPU token checks; action/RGB
    # arrays are not duplicated in the long-lived review result.
    result = dict(**identity, split=split, image=path.name,
                  label=raw["model_projection"], high_sample=record["high"],
                  pixel_means={k:float(v.mean()) for k,v in outer["pixel_values"].items()},
                  action_abs_max=float(outer["action"].abs().max()))
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU acceptance unexpectedly initialized CUDA")
    return result


def token_probe(arch, rows):
    """Use the existing proven tokenizer/field-mask test, not a text estimate."""
    import importlib.util
    import torch
    # Explicit reuse of the audited tokenizer preflight only; no benchmark
    # input pool, weight load, GPU loop, or timing estimate is invoked.
    path = Path(__file__).with_name("benchmark_memlite_high.py")
    spec = importlib.util.spec_from_file_location("high_token_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_pixels = {k: torch.empty(1, 3, 256, 256) for k in ("exterior", "wrist_left", "wrist_right")}
    fixtures = [dict(samples=row["high_sample"], pixel_values=fake_pixels) for row in rows]
    tokens, checks = module.preflight(arch, fixtures)
    return tokens, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16, choices=(8,16,32))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"images").mkdir()
    started = time.monotonic()
    manifest = json.loads((args.release/"manifest.json").read_text())
    selected = set()
    arrays = {s:np.load(args.release/f"{s}_candidates.npy", mmap_mode="r") for s in ("train","eval")}
    for split in ("train", "eval"):
        tasks = np.load(args.release/f"{split}_tasks.npy", mmap_mode="r")
        for task in range(100):
            ids = np.flatnonzero(tasks == task)
            for pos in (.31, .72):
                selected.add((split, int(ids[int((len(ids)-1)*pos)])))
    # Full compact scan: additional observed skills and the longest rendered
    # contexts. A source identity maps uniquely to its exact fixed-phase anchor.
    skill_coverage, longest = {}, []
    starts = {s: {} for s in arrays}
    for split, arr in arrays.items():
        changed = np.r_[True, arr[1:,0] != arr[:-1,0]]
        for index in np.flatnonzero(changed):
            starts[split][int(arr[index,0])] = int(index)
    with (args.release/"episodes.jsonl").open() as stream:
        for serial, line in enumerate(stream):
            ep = json.loads(line)
            if not ep["segments"]:
                continue
            split = ep["split"]
            anchors = anchor_records(ep["segments"], ep["phase"])
            seen = set()
            for anchor, (frame, segment, previous, parent, history) in enumerate(anchors):
                seg = ep["segments"][segment]
                index = starts[split][serial] + anchor
                for skill in seg["skills"]:
                    if skill["verb"] not in skill_coverage:
                        skill_coverage[skill["verb"]] = (split, index)
                key = (segment, previous, parent, tuple(history))
                if key in seen:
                    continue
                seen.add(key)
                p = projection(seg, branch="high", task_name=ep["task_name"], previous_intent=previous,
                               previous_parent=parent, history=history)
                size = sum(len(str(v)) for v in p.values())
                entry = (size, split, index)
                if len(longest) < 32:
                    heapq.heappush(longest, entry)
                elif entry > longest[0]:
                    heapq.heapreplace(longest, entry)
    selected.update(skill_coverage.values())
    selected.update((split,index) for _,split,index in longest)
    jobs = [(split,index,str(args.output),i) for i,(split,index) in enumerate(sorted(selected))]
    atomic_json(args.output/"selection.json",dict(jobs=jobs, skills=skill_coverage,longest=sorted(longest,reverse=True)))
    print(json.dumps(dict(stage="selection", windows=len(jobs), seconds=time.monotonic()-started)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(args.release),str(args.root))) as pool:
        results = list(pool.map(read_job, jobs, chunksize=1))
    arch = configuration(args.root,"high",manifest["task_names"])["arch"]
    tokens, checks = token_probe(arch, results)
    for row, token in zip(results,tokens):
        del row["high_sample"]
        row["tokens"] = token
    atomic_json(args.output/"rows.json",results)
    report = dict(status="CPU_PASS_REQUIRES_HUMAN_REVIEW", manifest_sha256=sha256(args.release/"manifest.json"),
                  rows_sha256=sha256(args.output/"rows.json"), windows=len(results), skills=sorted(skill_coverage),
                  tasks=sorted({r["task"] for r in results}),
                  max_checked_tokens=max(t["sequence_tokens"] for t in tokens), mixed_token_checks=checks,
                  seconds=time.monotonic()-started, cuda_initialized=False,
                  human_review_passed=False)
    atomic_json(args.output/"result.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()

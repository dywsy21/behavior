"""Publish immutable data acceptance only after recorded evidence passes.

This does not start training, alter raw data, or certify simulator success.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from g05.utils.training.stage1_runtime import atomic_json, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--bounds", type=Path, required=True)
    ap.add_argument("--owner-review", type=Path, required=True)
    args = ap.parse_args()
    destination = args.release/"acceptance.json"
    require(not destination.exists(), "Acceptance already exists; do not overwrite a release")
    manifest_sha = sha256(args.release/"manifest.json")
    manifest = json.loads((args.release/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        require(Path(name).name == name and sha256(args.release/name) == digest, "Compact file mismatch: "+name)
    paths = dict(source=args.evidence/"source-hash-v1/result.json", qa=args.evidence/"data-qa-v3/result.json",
                 normalizer=args.evidence/"normalizer-v2.json", sampler=args.evidence/"sampler-v4.json",
                 bounds=args.bounds/"receipt.json", review=args.owner_review)
    receipts = {k:json.loads(p.read_text()) for k,p in paths.items()}
    source,qa,norm,sampler,bounds,review = (receipts[k] for k in ("source","qa","normalizer","sampler","bounds","review"))
    require(source["status"] == "PASS", "Source content hashes failed")
    for key in ("qa","normalizer","sampler","bounds","review"):
        require(receipts[key]["manifest_sha256"] == manifest_sha, key+" uses a different data version")
    require(qa["status"] == "CPU_PASS_REQUIRES_HUMAN_REVIEW" and qa["tasks"] == list(range(100))
            and len(qa["skills"]) == 35 and qa["windows"] >= 400, "Insufficient actual all-task loader coverage")
    require(sha256(args.evidence/"data-qa-v3/rows.json") == qa["rows_sha256"] == review["qa_rows_sha256"]
            == norm["qa_rows_sha256"], "QA source rows mismatch")
    require(review["status"] == "OWNER_STRATIFIED_REVIEW_PASS" and review["current_unseen_windows"] == 0
            and review["current_selected_windows_seen"] == review["current_selected_windows"] >= 100,
            "Owner visual review missing")
    require(norm["status"] == "PASS" and norm["failures"] == 0 and norm["max_error"] <= 1e-4
            and norm["raw_state_anchor"] and norm["actual_transform_equals_streaming_bounds"], "Normalizer roundtrip failed")
    require(bounds["status"] == "TRAIN_BOUNDS_COMPLETE" and bounds["no_eval_contribution"]
            and bounds["normalized_coordinates_preserved"]
            and sha256(args.bounds/"stats.json") == bounds["stats_sha256"] == norm["stats_sha256"], "TRAIN statistics mismatch")
    require(sampler["status"] == "PASS", "Sampler audit failed")
    for branch in ("high","low"):
        r=sampler["recipes"][branch]
        require(r["duplicates"] == r["omitted"] == 0 and r["minimum_tasks_per_microbatch"] >= 2, "Sampling contract failed")
    provenance=json.loads((args.release/"split_provenance.json").read_text())
    protected=provenance["protected_episodes"]
    seen_protected, train_episodes, tasks, groups = set(), set(), set(), {}
    counts={s:0 for s in ("train","eval")}
    with (args.release/"episodes.jsonl").open() as stream:
        for line in stream:
            ep=json.loads(line);row=ep["row"];serial=int(row["episode_index"])
            group=(int(row["task_index"]),int(row["task_instance_id"]))
            require(group[1] < 301, "Public-test contamination")
            require(groups.setdefault(group,ep["split"]) == ep["split"], "Instance split leakage")
            if str(serial) in protected:
                require(protected[str(serial)] == ep["split"], "Old protected split changed")
                seen_protected.add(str(serial))
            if ep["segments"]:
                counts[ep["split"]]+=1;tasks.add(group[0])
                if ep["split"] == "train":train_episodes.add(serial)
    require(seen_protected == set(protected), "Old source disappeared")
    require(train_episodes == set(bounds["episodes"]), "Bounds include non-TRAIN or omit eligible source")
    require(tasks == set(range(100)), "Task omitted")
    require(manifest["clock"] == "published local frame_index; offset=0; overflow quarantined", "Clock rule changed")
    for split in ("train","eval"):
        n=len(np.load(args.release/f"{split}_candidates.npy",mmap_mode="r"))
        require(n == sum(r[split+"_anchors"] for r in manifest["task_counts"].values()), "Candidate counts mismatch")
    result=dict(status="ACCEPTED",accepted_at=datetime.now(timezone.utc).isoformat(),manifest_sha256=manifest_sha,
        gates={k:True for k in ("source_identity","protected_split","clock_alignment","human_review","normalizer","all_task_loader")},
        evidence={k:dict(path=str(p),sha256=sha256(p)) for k,p in paths.items()},eligible_episodes=counts,
        stats_sha256=norm["stats_sha256"],formal_training_authorized_by_this_receipt=False,
        limitations=review["limitations"])
    atomic_json(destination,result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()

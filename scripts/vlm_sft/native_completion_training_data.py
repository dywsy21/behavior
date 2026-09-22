"""Pinned reviewed TRAIN39 admission and deterministic 4+2+2 sampling.

Offline evidence is checked here, never projected into model queries. No
heldout, additional labels, duplicate terminal motion or synthetic HOLD.
"""
import json
from pathlib import Path
import random

from common import sha
from native_completion import validate_sidecar
from native_completion_dataset import DATASET_SHA as MOTION_DATA_SHA, ROWS_SHA as MOTION_ROWS_SHA
from native_completion_protocol import query_row
from native_dataset import load_dataset, checked_images

DATA_SOURCE = "ccb16f2ee268504c9bffde37179b4a06abcdf6e4"
DATA_SHA = "085d0a06eb590d0f390337951ca47ba412009010e69cd51a8665d838e8aba0fd"
STATES_SHA = "2fd27d707ad86e7fd6c09662000e99af26c0b1d686bbe9ff92b3c6aa8604fa99"
REVIEW_SHA = "7fbb349f81a71d944696cad4ad0d58b4766e4bde9d1b7a014714bbee9f7bbd2f"
MANUAL_SHA = "005a8b0cb7d08da01054f3b559931f99419f769a3ca9a7d4d696ab346665b4d6"


def read(path): return json.loads(Path(path).read_text())


def load_reviewed(completion, motion, review):
    completion, motion, review = map(Path, (completion, motion, review))
    for path, digest in ((completion/"completion_dataset.json", DATA_SHA), (completion/"states.jsonl", STATES_SHA),
                         (completion/"manual_review.json", MANUAL_SHA), (review, REVIEW_SHA),
                         (motion/"dataset.json", MOTION_DATA_SHA), (motion/"train.jsonl", MOTION_ROWS_SHA)):
        if sha(path) != digest: raise ValueError("Immutable reviewed completion/motion bytes changed: "+str(path))
    manifest, receipt = read(completion/"completion_dataset.json"), read(review)
    if (manifest["source_commit"] != DATA_SOURCE or receipt["source_commit"] != DATA_SOURCE or
            receipt["owner"] != "Codex-parent" or receipt["states_sha256"] != STATES_SHA or
            receipt["dataset_sha256"] != DATA_SHA or receipt["request_verify_is_success"] is not False):
        raise ValueError("Independent data-review identity mismatch")
    original, original_manifest = load_dataset(motion, require_gate=True)
    states = [json.loads(line) for line in (completion/"states.jsonl").read_text().splitlines()]
    examples = project_examples(original, states)
    for state in states: checked_images(state)
    for source in manifest["sources"]:
        for path, expected in [(source["inventory"], source["inventory_sha256"]),
                               (source["parent_review"], source["parent_review_sha256"])] + [
                (entry["path"], entry["sha256"]) for entry in source["files"].values()]:
            if sha(Path(path)) != expected: raise ValueError("Original reviewed source evidence changed")
    return examples, {"dataset_sha256": DATA_SHA, "states_sha256": STATES_SHA, "review_sha256": REVIEW_SHA,
        "motion_dataset_sha256": MOTION_DATA_SHA, "motion_rows_sha256": MOTION_ROWS_SHA,
        "data_source_commit": DATA_SOURCE, "motion_examples": 34, "status_examples": 39,
        "whole_runs": 5, "original_coverage": original_manifest["coverage"]}


def project_examples(original, states):
    """Validate preserved34 motions; construct73 queries, not new motion labels."""
    if len(original) != 34 or len(states) != 39 or len({r["id"] for r in states}) != 39:
        raise ValueError("Exactly34 unique motions and39 unique status states")
    originals = {row["id"]: row for row in original}
    if len(originals) != 34: raise ValueError("Duplicate original motion")
    examples = []; matched = set(); counts = {}; captures = set()
    for state in states:
        validate_sidecar(state)
        group = state["provenance"]["run_inventory_sha256"]
        capture = (group, state["provenance"]["capture_sha256"])
        if capture in captures: raise ValueError("Duplicate completion-state capture")
        captures.add(capture)
        label = state["label"]; status = label["skill_status"]
        counts.setdefault(group, {"CONTINUE": 0, "REQUEST_VERIFY": 0})[status] += 1
        if status == "CONTINUE":
            key = state["source_motion_row_id"]
            if key in matched or key not in originals: raise ValueError("Missing/duplicate original motion mapping")
            matched.add(key); old = originals[key]
            if any(state[k] != old[k] for k in ("actor", "protocol", "text", "images", "provenance")) or old["target"] != label["motion"]:
                raise ValueError("Original motion state/target/history changed")
            examples.append({"id": key+"/motion", "category": "motion", "group": group, "state": state,
                "query": query_row(state["actor"], "motion", old["target"])})
        examples.append({"id": state["id"]+"/status", "category": status, "group": group, "state": state,
            "query": query_row(state["actor"], "status", status)})
    if matched != set(originals) or len(counts) != 5 or any(c["CONTINUE"] < 1 or c["REQUEST_VERIFY"] != 1 for c in counts.values()):
        raise ValueError("Full five-run continuation/one-terminal membership required")
    if len(examples) != 73: raise ValueError("Unexpected supervision count")
    return examples


class BalancedSampler:
    def __init__(self, examples, seed=41):
        self.random = random.Random(seed); self.groups = {}
        for category in ("motion", "CONTINUE", "REQUEST_VERIFY"):
            groups = {}
            for example in examples:
                if example["category"] == category: groups.setdefault(example["group"], []).append(example)
            if len(groups) != 5: raise ValueError("Each category must cover all five TRAIN trajectories")
            self.groups[category] = groups
    def update(self):
        batches = []
        for category in ("motion", "motion", "CONTINUE", "REQUEST_VERIFY"):
            groups = self.groups[category]; keys = sorted(groups)
            batches.append([self.random.choice(groups[self.random.choice(keys)]) for _ in range(2)])
        return batches

"""List original-split source instances whose initial intent fits a local teacher."""
import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.data.collect_memlite_recovery import TARGETS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads(args.source_index.read_text())
    first = {}
    columns = ["episode_index", "frame_index", "primitive_idx", "intent", "memlite_branch", "memory_input_corruption"]
    for batch in pq.ParquetFile(index["view_identity"]["sidecar_paths"][0]).iter_batches(columns=columns):
        selected = batch.filter(pc.and_kleene(pc.equal(batch["primitive_idx"], 0),
                    pc.and_kleene(pc.equal(batch["memlite_branch"], "high"),
                                  pc.equal(batch["memory_input_corruption"], "none"))))
        for row in selected.to_pylist():
            ep = row["episode_index"]
            if ep not in first or row["frame_index"] < first[ep]["frame_index"]:
                first[ep] = row
    result = {"train": {str(t): [] for t in range(5)}, "eval": {str(t): [] for t in range(5)}, "rows": []}
    root = Path(index["view_identity"]["dataset_roots"][0][0])
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "task_index", "task_instance_id"]).to_pylist():
            ep, task, instance = row["episode_index"], row["task_index"], row["task_instance_id"]
            label = first.get(ep)
            match = label is not None and TARGETS[task] in label["intent"]
            split = "train" if instance in index["train_source_instances"][str(task)] else "eval"
            if match:
                result[split][str(task)].append(instance)
            result["rows"].append({**row, "split": split, "teacher_target": TARGETS[task],
                                   "target_matches_initial_intent": match, "label": label})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({s: {t: ids[:30] for t, ids in result[s].items()} for s in ("train", "eval")}), flush=True)


if __name__ == "__main__":
    main()

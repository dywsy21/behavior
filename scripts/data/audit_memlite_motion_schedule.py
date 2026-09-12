"""Check the actual train view, all planned ranks, and optional delivered receipts."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from g05.data.memlite_recovery_dataset import RecoveryAugmentedDataset
from g05.utils.common.motion_sampler import ResumableDistributedMotionBatchSampler
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.processor_utils import build_processors, instantiate_dataset


KINDS = ("high_original", "high_recovery", "yaw_stop", "yaw_start_reverse", "gripper", "recovery", "other")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sampling-index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--run-dir", type=Path)
    args = ap.parse_args()
    if args.output.exists() or args.steps < 1:
        raise ValueError("Need a new audit output and positive steps")
    register_default_resolvers()
    cfg = OmegaConf.load(args.source_run / ".hydra/config.yaml")
    processors = build_processors(cfg)
    original = instantiate_dataset(cfg, is_training_set=True)
    original.set_processor(processors)
    dataset = RecoveryAugmentedDataset(original, args.manifest, processors)
    index = json.loads(args.sampling_index.read_text())
    tasks = np.full(len(dataset), -1, np.int8)
    kinds = np.full(len(dataset), -1, np.int8)
    for task in index["tasks"]:
        for kind in KINDS:
            code = KINDS.index(kind)
            if kind.startswith("high"):
                coordinates = task[kind]
                assert (kinds[coordinates] == -1).all()
                tasks[coordinates], kinds[coordinates] = task["task_id"], code
            else:
                for start, end in task["low_strata"][kind]:
                    assert (kinds[start:end] == -1).all()
                    tasks[start:end], kinds[start:end] = task["task_id"], code
    counts, owned, first_batches = Counter(), [], []
    for rank in range(4):
        sampler = ResumableDistributedMotionBatchSampler(dataset, sampling_index=args.sampling_index,
            batch_size=8, num_replicas=4, rank=rank, seed=17, high_fraction=.125)
        planned = list(itertools.islice(sampler, args.steps))
        if len(planned) != args.steps:
            raise ValueError("Requested schedule exceeds actual training epoch")
        seen = set()
        for n, batch in enumerate(planned):
            names = Counter(KINDS[kinds[i]] for i in batch)
            expected = {"yaw_stop": 1, "yaw_start_reverse": 1, "gripper": 1, "recovery": 1, "other": 3,
                        "high_recovery" if n % 4 == 3 else "high_original": 1}
            assert names == expected and len(batch) == len(set(batch)) == 8
            assert all(tasks[i] >= 0 and kinds[i] >= 0 for i in batch)
            seen.update(batch)
            counts.update((int(tasks[i]), KINDS[kinds[i]]) for i in batch)
        if args.run_dir:
            rows = [json.loads(line) for line in (args.run_dir / f"sample_receipts_rank{rank}.jsonl").read_text().splitlines()]
            if len(rows) != args.steps:
                raise ValueError(f"Expected exactly {args.steps} delivered batches on rank {rank}")
            for n, (row, batch) in enumerate(zip(rows, planned)):
                actual = row["samples"]
                assert row["step"] == n+1
                assert [r["requested_index"] for r in actual] == batch
                for r, i in zip(actual, batch):
                    assert r["branch"] == ("high" if KINDS[kinds[i]].startswith("high") else "low")
                    assert r["source_kind"] == ("recovery" if i >= dataset.base_length else "original")
                    assert r["locator"] and r["actual_raw_index"] >= 0
        owned.append(seen)
        first_batches.append(planned[:5])
    assert all(owned[a].isdisjoint(owned[b]) for a in range(4) for b in range(a))
    if args.steps % 20 == 0:
        for kind in KINDS:
            assert len({counts[t, kind] for t in range(5)}) == 1
    output = {"passed": True, "steps_per_rank": args.steps, "world_size": 4,
        "delivered_receipts_verified": args.run_dir is not None,
        "data_loaded_for_training": False, "optimizer_updates_by_this_audit": 0,
        "actual_view_length": len(dataset), "source_rank_disjoint": True,
        "unique_source_rows_by_rank": [len(x) for x in owned],
        "counts": {f"task{t}/{k}": n for (t,k),n in sorted(counts.items())},
        "first_five_batches_by_rank": first_batches}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k:v for k,v in output.items() if k != "first_five_batches_by_rank"}), flush=True)


if __name__ == "__main__":
    main()

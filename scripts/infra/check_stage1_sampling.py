"""Exhaustively audit both 8-rank recipes over the actual released indices."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from g05.utils.training.stage1_sampling import MixedTaskPass
from g05.utils.training.stage1_runtime import atomic_json, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = np.load(args.release/"train_tasks.npy",mmap_mode="r")
    results = {}
    for component, micro, accumulation in (("high",4,8),("low",32,1)):
        started = time.monotonic()
        schedule = MixedTaskPass(tasks,micro_batch=micro,accumulation=accumulation)
        results[component] = dict(**schedule.audit(),seconds=time.monotonic()-started)
    result = dict(status="PASS",manifest_sha256=sha256(args.release/"manifest.json"),recipes=results)
    atomic_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()

"""Launch an unchanged simulator/policy entry point with bounded CPU pools.

Set thread-library environment variables BEFORE importing Torch/OpenCV. Optional
CPU affinity is explicit and must be inside the inherited process affinity.
This never changes CUDA visibility, physics, camera settings or model precision.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

def positive_int(value):
    parsed = int(value)
    if parsed < 1 or parsed > 1024:
        raise argparse.ArgumentTypeError("thread count must be between 1 and 1024")
    return parsed

def parse_affinity(value):
    selected = set()
    for part in value.split(","):
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError("CPU affinity must contain CPU numbers or inclusive ranges")
            lo, hi = map(int, pieces)
            if not 0 <= lo <= hi <= 1048576:
                raise ValueError("invalid CPU affinity range")
            selected.update(range(lo, hi + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError("CPU numbers must be nonnegative")
            selected.add(cpu)
    if not selected:
        raise ValueError("empty CPU affinity")
    return selected

def configure(threads, interop_threads, blas_threads, affinity=None):
    if affinity is not None:
        selected = parse_affinity(affinity)
        allowed = os.sched_getaffinity(0)
        if not selected <= allowed:
            raise ValueError(f"CPU affinity requests unavailable CPUs: {sorted(selected - allowed)}")
        os.sched_setaffinity(0, selected)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(blas_threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["OPENCV_FOR_THREADS_NUM"] = "1"
    torch = importlib.import_module("torch")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    cv2 = importlib.import_module("cv2")
    cv2.setNumThreads(1)
    return {"torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "opencv_threads": cv2.getNumThreads(), "blas_threads": blas_threads,
            "cpu_affinity": sorted(os.sched_getaffinity(0))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=positive_int, required=True)
    parser.add_argument("--interop-threads", type=positive_int, default=1)
    parser.add_argument("--blas-threads", type=positive_int, default=1)
    parser.add_argument("--cpu-affinity")
    parser.add_argument("target", type=Path)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    if not target.is_file():
        parser.error("target must be an existing Python file")
    budget = configure(args.threads, args.interop_threads, args.blas_threads, args.cpu_affinity)
    print("MEMLITE_CPU_BUDGET " + json.dumps(budget, sort_keys=True), flush=True)
    sys.argv = [str(target), *args.target_args]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()

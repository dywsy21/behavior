"""Validate robo's four-A100 topology for an opt-in evaluation preset."""
import argparse
import csv
import json
import os
import re
import subprocess

def parse_profile(gpu_csv, topology, allowed_cpus):
    devices = list(csv.reader(gpu_csv.strip().splitlines(), skipinitialspace=True))
    if len(devices) != 4 or [int(row[0]) for row in devices] != list(range(4)):
        raise ValueError("The robo preset requires exactly four GPUs indexed 0..3")
    if any("A100" not in row[1] or int(row[2]) < 76000 for row in devices):
        raise ValueError("The robo preset requires four A100 80GB GPUs")
    text = re.sub(r"\x1b\[[0-9;]*m", "", topology)
    affinities = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"GPU[0-3]", fields[0]) and len(fields) >= 7 and "X" in fields[1:5]:
            cpu_text = fields[5]
            cpus = set()
            for part in cpu_text.split(","):
                bounds = part.split("-")
                lo = int(bounds[0]); hi = int(bounds[-1])
                if len(bounds) > 2 or not 0 <= lo <= hi <= 1048576:
                    raise ValueError("Invalid CPU affinity reported by nvidia-smi")
                cpus.update(range(lo, hi + 1))
            if not cpus or not cpus <= allowed_cpus:
                raise ValueError("GPU-local CPUs are unavailable to this process")
            affinities[int(fields[0][3:])] = (cpu_text, cpus)
    if set(affinities) != set(range(4)):
        raise ValueError("Missing GPU-local CPU affinity in nvidia-smi topology")
    for first in range(4):
        for second in range(first + 1, 4):
            if affinities[first][1] & affinities[second][1]:
                raise ValueError("The robo preset requires disjoint GPU-local CPU groups")
    return {"profile": "robo_a100_v1", "sim_torch_threads": 1,
            "policy_torch_threads": 2, "blas_threads": 1, "interop_threads": 1,
            "opencv_threads": 1, "physics": "stock_cpu_unchanged",
            "renderer": "stock_unchanged", "video_codec": "libx264_cpu",
            "gpus": [{"index": int(row[0]), "name": row[1], "memory_mib": int(row[2]),
                      "cpu_affinity": affinities[int(row[0])][0]} for row in devices]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinities-only", action="store_true")
    args = parser.parse_args()
    gpu_csv = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True)
    topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    profile = parse_profile(gpu_csv, topo, os.sched_getaffinity(0))
    if args.affinities_only:
        print("\n".join(row["cpu_affinity"] for row in profile["gpus"]))
    else:
        print(json.dumps(profile, indent=2))

if __name__ == "__main__":
    main()

"""H64: one CPU-only pinned robot USD export; never loads a task or SimulationApp."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
ASSET_SHA = "6029617cdd3aefce981428058a1c82cffe6af20ae61dc5be1da7334342c3bc52"
DEFINITION_SHA = "63f841cffd5c499102416a797a22fa5b4cbaf146539df7dde8a4a1053e2326f1"
SDK_ROOT = Path("/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim/extscache/"
                "omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311")


def checked_sha(path, expected, maximum_bytes):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("Bounded regular pinned asset file required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Robot source asset changed")
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--definition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CPU-only environment required")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Clean fixed source required")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Never overwrite an existing asset export")
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    checked_sha(args.asset, ASSET_SHA, 160 * 1024 * 1024)
    checked_sha(args.definition, DEFINITION_SHA, 16 * 1024)
    import yaml
    from pxr import Usd, UsdGeom, UsdPhysics, Work
    from semantic_robot.v2.finger_collision_asset import export_stage
    if tuple(Usd.GetVersion()) != (0, 24, 5):
        raise ValueError("Only the audited installed USD 0.24.5 SDK is permitted")
    sdk_modules = {m.__name__: str(Path(m.__file__).resolve(strict=True))
                   for m in (Usd, UsdGeom, UsdPhysics, Work)}
    if any(not Path(path).is_relative_to(SDK_ROOT.resolve(strict=True)) for path in sdk_modules.values()):
        raise ValueError("USD imports do not belong to the installed audited SDK")
    Work.SetConcurrencyLimit(2)
    definition = yaml.safe_load(args.definition.read_text())
    stage = Usd.Stage.Open(str(args.asset), load=Usd.Stage.LoadNone)
    if not stage:
        raise ValueError("Pinned robot USD could not be opened")
    for layer in stage.GetUsedLayers():
        if not layer.anonymous and Path(layer.realPath).resolve() != args.asset.resolve():
            raise ValueError("Unpinned external USD layer is not permitted")
    surfaces = export_stage(stage, definition, Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics)
    # Recheck the files read by USD/YAML, not just their pre-open identities.
    checked_sha(args.asset, ASSET_SHA, 160 * 1024 * 1024)
    checked_sha(args.definition, DEFINITION_SHA, 16 * 1024)
    if any(name in sys.modules for name in ("torch", "isaacsim", "omnigibson")):
        raise ValueError("CPU asset reader imported a prohibited model/simulator module")
    if time.monotonic() - started >= 60:
        raise TimeoutError("60 second asset-read budget")
    summary = {name: {"meshes": len(row["meshes"]),
                     "vertices": sum(len(m["vertices_link_m"]) for m in row["meshes"].values()),
                     "triangles": sum(len(m["triangles"]) for m in row["meshes"].values())}
               for name, row in surfaces["links"].items()}
    result = {"status": "completed", "source_commit": commit, "pid": os.getpid(),
              "asset_sha256": ASSET_SHA, "definition_sha256": DEFINITION_SHA,
              "usd_version": list(Usd.GetVersion()), "usd_module_paths": sdk_modules,
              "seconds": time.monotonic() - started,
              "model_calls": 0, "simulator_resets": 0, "training_steps": 0,
              "success_rate": None, "not_success_rate": True, "summary": summary, "surfaces": surfaces}
    raw = json.dumps(result, indent=2, allow_nan=False).encode()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Two MiB output budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({k: v for k, v in result.items() if k != "surfaces"}))


if __name__ == "__main__": main()

"""Explicit 2026-09-21 NVMe runtime profile; old disk gates stay unchanged.

Source, model and installed environment remain on SDA. OG still copies its
small kit/logo files inside EXP_PATH; this profile does NOT claim read-only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from native_reference_profile import ROOT as EXPERIMENT_ROOT

PROFILE="h09y-nvme-runtime-20260921-v1"
NVME=Path("/mnt/nvme_tmp")
SDA=Path("/mnt/sdc1")
RUNTIME_ROOT=NVME/"robodojo_vlm_runtime_20260921"
SPEC={"profile":PROFILE,"runtime_root":str(RUNTIME_ROOT),"runtime_max_MiB":16384,
      "nvme_min_free_GiB":80,"sda_min_free_GiB":32}
CACHE_SUBDIRS={"OMNIGIBSON_APPDATA_PATH":"omnigibson","TMPDIR":"tmp",
    "TMP":"tmp","TEMP":"tmp",
    "CUDA_CACHE_PATH":"cuda","__GL_SHADER_DISK_CACHE_PATH":"gl",
    "XDG_CACHE_HOME":"xdg","TORCH_HOME":"torch","TRITON_CACHE_DIR":"triton",
    "HF_HOME":"huggingface","HF_HUB_CACHE":"huggingface/hub","HUGGINGFACE_HUB_CACHE":"huggingface/hub",
    "HF_ASSETS_CACHE":"huggingface/assets","HF_DATASETS_CACHE":"huggingface/datasets",
    "TRANSFORMERS_CACHE":"huggingface/transformers","TORCH_EXTENSIONS_DIR":"torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR":"torchinductor"}


def validate_spec(release):
    if "storage" not in release:return False
    spec=release["storage"]
    if not isinstance(spec,dict) or set(spec)!=set(SPEC) or any(type(spec[k]) is not type(v) or spec[k]!=v for k,v in SPEC.items()):
        raise ValueError("Exact explicit NVMe runtime storage profile required")
    return True


def runtime_environment(output):
    output=Path(output).resolve()
    if output.parent!=EXPERIMENT_ROOT or not output.name:
        raise ValueError("Runtime cache belongs to one registered direct H09Y run")
    root=RUNTIME_ROOT/output.name
    return {**{key:str(root/name) for key,name in CACHE_SUBDIRS.items()},"PYTHONDONTWRITEBYTECODE":"1"}


def existing_parent(path):
    path=Path(path)
    while not path.exists():
        if path.is_symlink():raise ValueError("Broken runtime symlink")
        if path.parent==path:raise ValueError("No existing storage ancestor")
        path=path.parent
    return path


class RuntimeStorage:
    def __init__(self,release,output):
        if not validate_spec(release):raise ValueError("No explicit new storage profile")
        self.output=Path(output).resolve();self.expected=runtime_environment(self.output)
        self.root=RUNTIME_ROOT/self.output.name

    def paths(self):
        if not os.path.ismount(NVME):raise ValueError("NVMe path is not an actual mount")
        device=NVME.stat().st_dev
        for path in [EXPERIMENT_ROOT,RUNTIME_ROOT,self.root,*[Path(self.expected[k]) for k in CACHE_SUBDIRS]]:
            resolved=path.resolve()
            if not resolved.is_relative_to(NVME.resolve()) or existing_parent(resolved).stat().st_dev!=device:
                raise ValueError("Runtime/output resolves outside the actual NVMe filesystem")
        for key,value in self.expected.items():
            if os.environ.get(key)!=value:raise ValueError("Runtime environment differs: "+key)
        # The environment must already be set by the launcher, before Python
        # imports source files from SDA. This assignment also guards embedders.
        sys.dont_write_bytecode=True

    def check(self):
        self.paths()
        if shutil.disk_usage(NVME).free<80*1024**3 or shutil.disk_usage(SDA).free<32*1024**3:
            raise RuntimeError("Explicit NVMe80/SDA32 free-space gate")
        size=0
        for path in RUNTIME_ROOT.rglob("*"):
            if path.is_symlink():raise ValueError("Uncounted runtime cache symlink is forbidden")
            if path.is_file():
                if path.stat().st_dev!=NVME.stat().st_dev:raise ValueError("Runtime cache crosses a filesystem")
                size+=path.stat().st_size
        if size>=16384*1024**2:raise RuntimeError("Whole new runtime cache exceeds 16 GiB")
        return {"profile":PROFILE,"runtime_root":str(RUNTIME_ROOT),"run_runtime_root":str(self.root),
            "runtime_bytes":size,"environment":self.expected,"source_environment_fully_readonly":False,
            "known_sda_writes":"installed OG EXP_PATH kit/logo copies"}

    def create(self):
        receipt=self.check()
        for key in CACHE_SUBDIRS:Path(self.expected[key]).mkdir(parents=True,exist_ok=True)
        self.check();return receipt


def activate(release,output):
    """Called before evaluator/model imports; no silent default migration."""
    if not validate_spec(release):return None
    storage=RuntimeStorage(release,output);storage.create();return storage


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--check",action="store_true")
    a=p.parse_args()
    print(json.dumps(RuntimeStorage({"storage":SPEC},a.output).check() if a.check else runtime_environment(a.output),indent=2))

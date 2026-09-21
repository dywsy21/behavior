"""Explicit 2026-09-21 NVMe runtime profile; old disk gates stay unchanged.

Source, model and installed environment remain on SDA. OG still copies its
small kit/logo files inside EXP_PATH; this profile does NOT claim read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

from native_reference_profile import ROOT as EXPERIMENT_ROOT

PROFILE="h09y-nvme-runtime-20260921-v1"
NVME=Path("/mnt/nvme_tmp")
SDA=Path("/mnt/sdc1")
RUNTIME_ROOT=NVME/"robodojo_vlm_runtime_20260921"
SPEC={"profile":PROFILE,"runtime_root":str(RUNTIME_ROOT),"runtime_max_MiB":16384,
      "nvme_min_free_GiB":80,"sda_min_free_GiB":32}
SHARED_PROFILE="h09y-nvme-shared-og-cache-20260921-v2"
SHARED_TARGET=RUNTIME_ROOT/"reference_train114_v1/omnigibson/global/cache"
SHARED_OWNER=1003
SHARED_RUNS=tuple([f"native_t1_i{i}_p{p:04d}" for i,p in ((114,993),(192,392),(114,989),(192,388),(114,985))]+
    [f"eval_t1_i{i}_{v}_v1" for i in (1,71) for v in ("base","finetuned","proprio_history_nn")]+
    ["training_v1","service_v1","eval_prepare_i1_v1","eval_prepare_i71_v1"])
OG=Path("/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson")
SHARED_SOFTWARE={str(OG/"macros.py"):"5e8ebb1a00fa3e864c5a4984f126442e91b6653f5869977ecddbf4039ea73228",
    str(OG/"simulator.py"):"d800c2832f24962c440c4781ebfdb4ea78c74aac37d2c74ee7894ae430f1e2a9",
    str(OG/"omnigibson_5_1_0.kit"):"1aec3eda2c9841a060070c16305ea90c72c92a56cf0c973084b88f61b23f140d"}
SHARED_SPEC={**SPEC,"profile":SHARED_PROFILE,"shared_og_cache":{
    "canonical_target":str(SHARED_TARGET),"owner_uid":SHARED_OWNER,"owner_name":"robodojo",
    "allowed_aliases":[str(RUNTIME_ROOT/name/"omnigibson/global/cache") for name in SHARED_RUNS],
    "software_sha256":SHARED_SOFTWARE,"mutable_software_cache_not_evidence":True,
    "reference_result_sha256":"1208c9bc86c40bc89cdfaae96362a769c23e34e6ed237d2231ed05ddfb8d7e0c"}}
DIVERSE_STORAGE_PROFILE="h09y-nvme-shared-og-cache-diverse-20260921-v3"
DIVERSE_RUNS=SHARED_RUNS+("native_t1_i192_p0380","native_t1_i114_p0969")
DIVERSE_SPEC={**SHARED_SPEC,"profile":DIVERSE_STORAGE_PROFILE,"shared_og_cache":{
    **SHARED_SPEC["shared_og_cache"],
    "allowed_aliases":[str(RUNTIME_ROOT/name/"omnigibson/global/cache") for name in DIVERSE_RUNS]}}
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
    expected=({SHARED_PROFILE:SHARED_SPEC,DIVERSE_STORAGE_PROFILE:DIVERSE_SPEC}.get(spec.get("profile"),SPEC)
              if isinstance(spec,dict) else SPEC)
    # Canonical JSON also rejects bool-as-int inside nested shared bindings.
    if not isinstance(spec,dict) or json.dumps(spec,sort_keys=True)!=json.dumps(expected,sort_keys=True):
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


def tree_bytes(root,device,aliases=None):
    """Never use rglob's silent permission-error skipping for a hard cap."""
    root=Path(root)
    if not root.exists():return 0
    total=0;pending=[root];visited=set();aliases=aliases or {}
    while pending:
        folder=pending.pop();info=folder.lstat();mode=info.st_mode
        if info.st_dev!=device:raise ValueError("Runtime cache crosses a filesystem")
        if (not stat.S_ISDIR(mode) or not mode&0o444 or not mode&0o111 or
                not os.access(folder,os.R_OK|os.X_OK)):
            raise PermissionError("Unreadable runtime cache directory: "+str(folder))
        canonical=folder.resolve()
        if canonical in visited:continue
        visited.add(canonical)
        with os.scandir(folder) as entries:
            for entry in entries:
                info=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    link=Path(entry.path);target=aliases.get(link)
                    try:
                        allowed=(target is not None and Path(os.readlink(link))==target and target.resolve(strict=True)==target and
                            target.is_relative_to(root.resolve()) and link.resolve(strict=True)==target and
                            stat.S_ISDIR(target.lstat().st_mode) and info.st_uid==target.lstat().st_uid)
                    except (OSError,RuntimeError):allowed=False
                    if not allowed:
                        raise ValueError("Unregistered, looping or escaped runtime cache alias")
                    pending.append(target);continue
                if info.st_dev!=device:raise ValueError("Runtime cache crosses a filesystem")
                if stat.S_ISDIR(info.st_mode):pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    if not info.st_mode&0o444 or not os.access(entry.path,os.R_OK):raise PermissionError(entry.path)
                    total+=info.st_size
                else:raise ValueError("Unexpected special runtime cache entry: "+entry.path)
    return total


class RuntimeStorage:
    def __init__(self,release,output):
        if not validate_spec(release):raise ValueError("No explicit new storage profile")
        self.output=Path(output).resolve();self.expected=runtime_environment(self.output)
        self.root=RUNTIME_ROOT/self.output.name
        self.profile=release["storage"]["profile"]
        self.shared=self.profile in (SHARED_PROFILE,DIVERSE_STORAGE_PROFILE)
        runs=DIVERSE_RUNS if self.profile==DIVERSE_STORAGE_PROFILE else SHARED_RUNS
        if self.shared and self.output.name not in runs:raise ValueError("Unregistered shared-cache run")
        self.aliases={RUNTIME_ROOT/name/"omnigibson/global/cache":SHARED_TARGET for name in runs} if self.shared else {}

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
        if self.shared:
            import pwd
            info=SHARED_TARGET.lstat()
            if (SHARED_TARGET.resolve(strict=True)!=SHARED_TARGET or not stat.S_ISDIR(info.st_mode) or
                    info.st_uid!=SHARED_OWNER or pwd.getpwuid(info.st_uid).pw_name!="robodojo" or
                    info.st_dev!=NVME.stat().st_dev):
                raise ValueError("Exact same-NVMe robodojo-owned completed reference cache required")
            for path,expected in SHARED_SOFTWARE.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError("Shared cache software changed")
            reference=EXPERIMENT_ROOT/"reference_train114_v1"
            if hashlib.sha256((reference/"result.json").read_bytes()).hexdigest()!=SHARED_SPEC["shared_og_cache"]["reference_result_sha256"]:
                raise ValueError("Completed reference identity changed")
            pid=json.loads(reference.with_suffix(".launch.json").read_text())["pid"]
            if type(pid) is not int or pid<=0 or Path(f"/proc/{pid}").exists():
                raise ValueError("Original reference process must have exited before software-cache reuse")
        if shutil.disk_usage(NVME).free<80*1024**3 or shutil.disk_usage(SDA).free<32*1024**3:
            raise RuntimeError("Explicit NVMe80/SDA32 free-space gate")
        size=tree_bytes(RUNTIME_ROOT,NVME.stat().st_dev,self.aliases)
        if size>=16384*1024**2:raise RuntimeError("Whole new runtime cache exceeds 16 GiB")
        return {"profile":self.profile,"runtime_root":str(RUNTIME_ROOT),"run_runtime_root":str(self.root),
            "runtime_bytes":size,"environment":self.expected,"source_environment_fully_readonly":False,
            "mutable_shared_software_cache":str(SHARED_TARGET) if self.shared else None,
            "shared_aliases":{str(k):str(v) for k,v in self.aliases.items()},
            "known_sda_writes":"installed OG EXP_PATH kit/logo copies"}

    def create(self):
        receipt=self.check()
        for key in CACHE_SUBDIRS:Path(self.expected[key]).mkdir(parents=True,exist_ok=True)
        if self.shared:
            alias=self.root/"omnigibson/global/cache";alias.parent.mkdir(parents=True,exist_ok=True)
            if not alias.is_symlink():
                if alias.exists():raise ValueError("Never replace an existing cache with a shared alias")
                alias.symlink_to(SHARED_TARGET,target_is_directory=True)
        # Kit creates this empty directory as mode000 on this installation.
        # Create it readably in the NEW per-run root before importing Kit;
        # never chmod an existing path or modify an old cache automatically.
        (self.root/"omnigibson/local/data/documents/Kit/shared/screenshots").mkdir(mode=0o700,parents=True,exist_ok=True)
        self.check();return receipt


def activate(release,output):
    """Called before evaluator/model imports; no silent default migration."""
    if not validate_spec(release):return None
    storage=RuntimeStorage(release,output);storage.create();return storage


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--check",action="store_true")
    a=p.parse_args()
    print(json.dumps(RuntimeStorage({"storage":SPEC},a.output).check() if a.check else runtime_environment(a.output),indent=2))

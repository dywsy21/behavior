"""Per-run cache placement; never change HOME or shared environments."""
from pathlib import Path


def isolated_cache_env(control):
    control=Path(control).resolve()
    if not control.is_dir():
        raise ValueError('Create and lock the run supervisor before allocating caches')
    names={'TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor',
           'TORCH_HOME':'torch','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg',
           'TMPDIR':'tmp'}
    result={}
    for key,name in names.items():
        path=control/'runtime-cache'/name
        if not path.resolve().is_relative_to(control):
            raise ValueError('Run cache may not resolve outside its supervisor')
        path.mkdir(parents=True,exist_ok=True)
        result[key]=str(path)
    return result

"""Per-run cache placement; never change HOME or shared environments."""
from pathlib import Path


CACHE_NAMES={'TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor',
             'TORCH_HOME':'torch','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg',
             'TMPDIR':'tmp'}


def isolated_cache_env(control):
    control=Path(control).resolve()
    if not control.is_dir():
        raise ValueError('Create and lock the run supervisor before allocating caches')
    result={}
    for key,name in CACHE_NAMES.items():
        path=control/'runtime-cache'/name
        if not path.resolve().is_relative_to(control):
            raise ValueError('Run cache may not resolve outside its supervisor')
        path.mkdir(parents=True,exist_ok=True)
        result[key]=str(path)
    return result


def isolated_rank_cache_env(cache_env,local_rank):
    """Partition the already-owned run caches before importing CUDA kernels.

    In particular, concurrent cold Triton driver builds must not replace the
    same cuda_utils.so over NFS. No process-global HOME or shared env mutation.
    """
    if type(local_rank) is not int or local_rank<0:
        raise ValueError('Expected a nonnegative integer local rank')
    result={}
    for key in CACHE_NAMES:
        if not cache_env.get(key):
            raise ValueError('Missing run-owned cache: '+key)
        base=Path(cache_env[key])
        if not base.is_absolute() or not base.is_dir():
            raise ValueError('Rank cache requires an existing absolute run cache')
        base=base.resolve();path=base/f'rank-{local_rank:03d}'
        if not path.resolve().is_relative_to(base):
            raise ValueError('Rank cache may not escape its run cache')
        path.mkdir(exist_ok=True);result[key]=str(path)
    return result

"""Fail before model loading if Triton/FLA selected an incompatible backend."""
import importlib


def initialize_cuda_backend(local_rank,*,import_module=importlib.import_module):
    # FLA caches backend detection at import and swallows the original driver
    # exception. Initialize the driver explicitly first so an error remains
    # actionable, rather than becoming a late torch.cpu.device AttributeError.
    torch=import_module('torch')
    if not torch.cuda.is_available():
        raise RuntimeError('Recovery training requires CUDA; no CPU fallback')
    torch.cuda.set_device(local_rank)
    triton=import_module('triton')
    target=triton.runtime.driver.active.get_current_target()
    if target.backend!='cuda':
        raise RuntimeError('Recovery training expected Triton CUDA, got '+str(target))
    fla=import_module('fla.utils')
    if (fla.get_available_device()!='cuda' or fla.device_torch_lib is not torch.cuda
            or not fla.IS_NVIDIA):
        raise RuntimeError('FLA has a stale/non-CUDA backend; restart with isolated rank caches')
    with fla.custom_device_ctx(local_rank):
        if torch.cuda.current_device()!=local_rank:
            raise RuntimeError('FLA device context does not match the local rank')
    return dict(local_rank=local_rank,torch_version=torch.__version__,
                triton_version=triton.__version__,backend=target.backend,
                arch=target.arch,device_name=torch.cuda.get_device_name(local_rank))

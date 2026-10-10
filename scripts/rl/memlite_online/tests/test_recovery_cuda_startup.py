from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_cuda_startup import initialize_cuda_backend


class CudaStartupTests(unittest.TestCase):
    def setup_modules(self):
        calls=[]
        cuda=NS(is_available=lambda:True,set_device=lambda r:calls.append(('device',r)),
                current_device=lambda:3,get_device_name=lambda r:'test-device')
        torch=NS(cuda=cuda,__version__='mock')
        def target():
            calls.append('target');return NS(backend='cuda',arch=80)
        triton=NS(runtime=NS(driver=NS(active=NS(get_current_target=target))),__version__='mock')
        fla=NS(get_available_device=lambda:'cuda',device_torch_lib=cuda,IS_NVIDIA=True,
               custom_device_ctx=lambda rank:nullcontext())
        modules={'torch':torch,'triton':triton,'fla.utils':fla}
        def importer(name): calls.append(name);return modules[name]
        return calls,modules,importer

    def test_driver_before_fla_and_correct_rank(self):
        calls,_,importer=self.setup_modules()
        result=initialize_cuda_backend(3,import_module=importer)
        self.assertEqual(result['local_rank'],3)
        self.assertLess(calls.index(('device',3)),calls.index('target'))
        self.assertLess(calls.index('target'),calls.index('fla.utils'))

    def test_original_driver_failure_is_not_swallowed(self):
        calls,modules,importer=self.setup_modules()
        def fail(): raise OSError('cold cache original exception')
        modules['triton'].runtime.driver.active.get_current_target=fail
        with self.assertRaisesRegex(OSError,'original exception'):
            initialize_cuda_backend(3,import_module=importer)
        self.assertNotIn('fla.utils',calls)

    def test_stale_backend_and_wrong_rank_rejected(self):
        _,modules,importer=self.setup_modules()
        modules['fla.utils'].device_torch_lib=object()
        with self.assertRaisesRegex(RuntimeError,'stale'):
            initialize_cuda_backend(3,import_module=importer)
        _,modules,importer=self.setup_modules()
        with self.assertRaisesRegex(RuntimeError,'local rank'):
            initialize_cuda_backend(2,import_module=importer)

    def test_no_cuda_or_non_nvidia_no_fallback(self):
        calls,modules,importer=self.setup_modules()
        modules['torch'].cuda.is_available=lambda:False
        with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):
            initialize_cuda_backend(3,import_module=importer)
        self.assertNotIn('fla.utils',calls)
        calls,modules,importer=self.setup_modules()
        modules['triton'].runtime.driver.active.get_current_target=lambda:NS(backend='cpu')
        with self.assertRaisesRegex(RuntimeError,'expected Triton CUDA'):
            initialize_cuda_backend(3,import_module=importer)
        self.assertNotIn('fla.utils',calls)


if __name__=='__main__':unittest.main()

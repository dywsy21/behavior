from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from g05.rl.native_decode import cpu_postprocess_inputs


class NativeDecodeTests(unittest.TestCase):
    def check_boundary(self, device):
        action=torch.ones(1,32,27,device=device,requires_grad=True)
        batch={'proprio':torch.zeros(1,1,27,device=device),
               'action_dim_is_pad':torch.zeros(1,27,device=device,dtype=torch.bool),
               'proprio_dim_is_pad':torch.zeros(1,27,device=device,dtype=torch.bool),
               'action_op_mask':torch.ones(1,32,27,device=device,dtype=torch.bool),
               'pixel_values':object(),'selected_action_source':'ar'}
        result=cpu_postprocess_inputs(action,batch)
        self.assertEqual(result['selected_action_source'],'fm')
        self.assertNotIn('pixel_values',result)
        self.assertEqual(batch['selected_action_source'],'ar')
        self.assertNotIn('action',batch)
        for key,value in result.items():
            if torch.is_tensor(value):
                self.assertEqual(value.device.type,'cpu')
                self.assertFalse(value.requires_grad)
        self.assertTrue(torch.equal(result['action'],action.detach().cpu()))

    def test_native_cpu_boundary_and_no_source_mutation(self):
        self.check_boundary('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'GPU boundary requires CUDA')
    def test_cuda_outputs_and_all_masks_return_to_cpu(self):
        self.check_boundary('cuda:0')


if __name__=='__main__': unittest.main()

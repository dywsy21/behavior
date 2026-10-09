import importlib.util
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[4]


def helper(name):
    spec=importlib.util.spec_from_file_location('_recovery_'+name,REPO/'src/g05/models/g05/helpers'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class ObserverAdapterTests(unittest.TestCase):
    def test_dedicated_adapter_boundary_rejects_planner_or_low_gradients(self):
        import torch
        from types import SimpleNamespace
        check=helper('observer_adapter').require_observer_adapter_only
        def policy(names):
            return SimpleNamespace(named_parameters=lambda:[(n,torch.nn.Parameter(torch.ones(1))) for n in names])
        names=['model.vlm.base_model.model.layers.0.q_proj.lora_A.outcome_observer.weight']
        self.assertEqual(check(policy(names)),names)
        for bad in ['model.vlm.layers.0.q_proj.weight','model.action_expert.q_proj.weight',
                    'model.vision_tower.weight','outcome_head.classifier.weight',
                    names[0].replace('outcome_observer','skill_fm')]:
            with self.assertRaises(ValueError):check(policy(names+[bad]))
        with self.assertRaises(ValueError):check(policy([]))

    def test_temporal_context_grad_opt_in_leaves_proprio_detached(self):
        import torch
        # Reuse the existing light helper loader without importing a VLM.
        sys.path.insert(0,str(Path(__file__).parent))
        from test_recovery_pipeline import module
        head=module.TemporalOutcomeObserver(16,width=8,include_absolute_proprio=True)
        x=torch.randn(2,2,16,requires_grad=True);p=torch.randn(2,2,27,requires_grad=True)
        steps=torch.tensor([[0,16],[0,16]]);mask=torch.ones(2,2,dtype=torch.bool)
        head(x,p,steps,mask).square().mean().backward()
        self.assertIsNone(x.grad);self.assertIsNone(p.grad)
        head.zero_grad(set_to_none=True)
        head(x,p,steps,mask,allow_context_grad=True).square().mean().backward()
        self.assertIsNotNone(x.grad);self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertIsNone(p.grad)


if __name__=='__main__':unittest.main()

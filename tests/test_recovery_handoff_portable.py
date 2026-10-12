import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_portable import portable_arch, materialize_arch, check_prefix_rows


class PortableTests(unittest.TestCase):
    def test_only_two_asset_paths_are_relocated_not_model_settings(self):
        arch=dict(hf_processor_path='/source/models/qwen3_5_2b_base_processor',
                  AT_CONFIG=dict(ckpt_dir='/source/models/action_tokenizer.pt'),
                  planner_outcome=dict(memory_update_ce_weight=.25),num_obs_steps=1)
        before=copy.deepcopy(arch)
        portable=portable_arch(arch,Path('/source'))
        actual=materialize_arch(portable,Path('/dest'))
        self.assertEqual(arch,before)
        self.assertEqual(actual['planner_outcome'],arch['planner_outcome'])
        self.assertEqual(actual['hf_processor_path'],'/dest/assets/hf')
        self.assertEqual(actual['AT_CONFIG']['ckpt_dir'],'/dest/assets/action_tokenizer.pt')

    def test_foreign_or_additional_absolute_dependency_rejected(self):
        arch=dict(hf_processor_path='/source/models/qwen3_5_2b_base_processor',
                  AT_CONFIG=dict(ckpt_dir='/source/models/action_tokenizer.pt'))
        changed=copy.deepcopy(arch);changed['hf_processor_path']='/outside/processor'
        with self.assertRaises(ValueError):portable_arch(changed,'/source')
        changed=copy.deepcopy(arch);changed['extra']={'path':'/another/weight.pt'}
        with self.assertRaises(ValueError):portable_arch(changed,'/source')
        portable=portable_arch(arch,'/source');portable['AT_CONFIG']['ckpt_dir']='../weights'
        with self.assertRaises(ValueError):materialize_arch(portable,'/dest')

    def test_full_original_cpu_prefixes_only_no_partial_filter_or_eval(self):
        rows=[dict(sample_id=str(i),pixels_sha256=str(i)) for i in range(383)]
        accepted=dict(status='passed',data_split='train',optimizer_updates=0,rows=copy.deepcopy(rows))
        check_prefix_rows(rows,accepted)
        for change in ('missing','reorder','alter','duplicate','eval','optimizer'):
            values=copy.deepcopy(rows);reference=copy.deepcopy(accepted)
            if change=='missing':values.pop()
            if change=='reorder':values.reverse()
            if change=='alter':values[0]['pixels_sha256']='changed'
            if change=='duplicate':values[0]=values[1]
            if change=='eval':reference['data_split']='eval'
            if change=='optimizer':reference['optimizer_updates']=1
            with self.assertRaises(ValueError):check_prefix_rows(values,reference)


if __name__=='__main__':unittest.main()

import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'scripts/experiments/eval_g05_100k.py'
spec = importlib.util.spec_from_file_location('bounded_g05_eval', PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class BudgetTests(unittest.TestCase):
    def test_fixed_windows_keep_actual_episode_ids(self):
        rows=evaluation.choose_windows([0,100],[100,300],[199,399])
        self.assertEqual([r['episode'] for r in rows],[199,199,399,399])
        self.assertEqual([r['index'] for r in rows],[17,51,142,226])
        self.assertTrue(all(r['index']+32<=300 for r in rows))

    def test_short_or_misaligned_episodes_fail(self):
        with self.assertRaises(ValueError): evaluation.choose_windows([0],[20],[199])
        with self.assertRaises(ValueError): evaluation.choose_windows([0,10],[100],[199])

    def test_budget_fits_action_call_cap(self):
        self.assertEqual(sum(evaluation.LIMITS),7320)
        self.assertEqual(sum((n+15)//16 for n in evaluation.LIMITS),458)

    def test_only_declared_checkpoints(self):
        with self.assertRaises(ValueError): evaluation.checkpoint(5000)
        self.assertEqual(evaluation.checkpoint(100000).name,'step_100000.pt')

    def test_vector_order_and_no_missing_base(self):
        import numpy as np
        keys=('base_qvel','trunk_qpos','left_arm','left_gripper','right_arm','right_gripper')
        dims=(3,4,7,1,7,1)
        parts={k:np.full((1,32,d),i,dtype=np.float32) for i,(k,d) in enumerate(zip(keys,dims))}
        result=evaluation.vector_chunk(parts)
        self.assertEqual(result.shape,(32,23))
        self.assertEqual(result[0].tolist(),[0]*3+[1]*4+[2]*7+[3]+[4]*7+[5])
        del parts['base_qvel']
        with self.assertRaises(KeyError): evaluation.vector_chunk(parts)

    def test_nonfinite_action_rejected(self):
        import numpy as np
        parts={k:np.zeros((32,d),dtype=np.float32) for k,d in zip(
            ('base_qvel','trunk_qpos','left_arm','left_gripper','right_arm','right_gripper'),(3,4,7,1,7,1))}
        parts['base_qvel'][0,0]=np.nan
        with self.assertRaises(ValueError): evaluation.vector_chunk(parts)


if __name__=='__main__': unittest.main()

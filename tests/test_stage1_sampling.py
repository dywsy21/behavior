import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("stage1_sampling", Path(__file__).resolve().parents[1] /
    "src/g05/utils/training/stage1_sampling.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExactMixedPassTests(unittest.TestCase):
    def test_all_tail_sizes_both_recipes(self):
        for remainder in range(256):
            tasks = np.arange(512 + remainder) % 100
            for micro, accum in ((4, 8), (32, 1)):
                schedule = m.MixedTaskPass(tasks, micro_batch=micro, accumulation=accum)
                audit = schedule.audit()
                self.assertEqual(audit["omitted"], 0)
                self.assertEqual(audit["candidates"], len(tasks))

    def test_reproducible_resume_and_epoch_shuffle(self):
        tasks = np.arange(1001) % 17
        a, b = m.MixedTaskPass(tasks), m.MixedTaskPass(tasks)
        self.assertEqual(a.digest, b.digest)
        self.assertNotEqual(a.digest, m.MixedTaskPass(tasks, epoch=1).digest)
        for rank in range(8):
            full = list(m.CommittedBatchSampler(a, rank))
            resumed = list(m.CommittedBatchSampler(b, rank, 2))
            self.assertEqual(full[16:], resumed)

    def test_task_repair_no_resampling(self):
        schedule = m.MixedTaskPass(np.repeat([0, 1], 512))
        self.assertGreaterEqual(schedule.audit()["minimum_tasks_per_microbatch"], 2)

    def test_impossible_recipe_fails(self):
        with self.assertRaises(ValueError):
            m.MixedTaskPass(np.r_[np.zeros(500), np.ones(12)])


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('learning_curve',Path(__file__).resolve().parents[1]/'scripts/rl/plot_learning_curve.py')
curve=importlib.util.module_from_spec(spec);spec.loader.exec_module(curve)


class LearningCurveTests(unittest.TestCase):
    def test_pre_update_alignment_and_no_smoothing_across_difficulty_changes(self):
        batches=[]
        for i,prefix in enumerate([100,100,4]):
            batches.append(dict(batch=i,run='test',new_actor_updates=4,
                episodes=[dict(worker=w,instance=w+1,prefix=prefix,success=i<2) for w in (0,1)]))
        points,actor=curve.training_points(dict(first_actor_updates=94,batches=batches))
        self.assertEqual(actor,106)
        self.assertEqual([p['actor_updates'] for p in points[::2]],[94,98,102])
        segments=curve.curriculum_segments(points,0)
        self.assertEqual([len(s) for s in segments],[2,1])
        self.assertEqual(segments[1][0]['rolling_rate'],0)
        self.assertEqual(segments[1][0]['rolling_n'],1)
        with self.assertRaises(ValueError):curve.training_points(dict(first_actor_updates=94,batches=batches[1:]))

    def test_incomplete_fixed_evaluation_is_not_a_zero_success_rate(self):
        with self.assertRaises(ValueError):curve.fixed_results(dict(evals=[dict(stage='E3',actor_updates=200,rows=[])]))


if __name__=='__main__':unittest.main()

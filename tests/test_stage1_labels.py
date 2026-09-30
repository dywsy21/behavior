import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g05.data.memlite_stage1_labels import compile_episode, anchor_records, projection, fixed_phase
from g05.data_processor.processor.memlite_v6_projection import _model_safe_v6_label


def raw_skill(index, start, end, sid=1, desc="move to", objects=None):
    return dict(skill_idx=index, skill_id=[sid], skill_description=[desc],
                object_id=[objects or ["radio"]], manipulating_object_id=[],
                spatial_prefix=[], memory_prefix=["back"], frame_duration=[start, end])


class Stage1LabelsTests(unittest.TestCase):
    def test_parallel_start_cuts_horizon_and_prior_is_causal(self):
        annotation = dict(meta_data=dict(valid_duration=[0, 100]), skill_annotation=[
            raw_skill(0, 0, 100), raw_skill(1, 40, 80, 8, "release")])
        segments = compile_episode(annotation, dict(length=110), "turn on radio")
        self.assertEqual([(s["start"], s["end"]) for s in segments], [(0, 40), (40, 80), (80, 100)])
        anchors = anchor_records(segments, 1)
        self.assertEqual([a[0] for a in anchors], list(range(1, 100, 16)))
        self.assertEqual(anchors[0][2:4], ("None", "None"))
        self.assertEqual(anchors[3][2], segments[0]["text"])
        for frame, segment, prev, parent, history in anchors:
            p = projection(segments[segment], branch="high", task_name="turn on radio",
                           previous_intent=prev, previous_parent=parent, history=history)
            _model_safe_v6_label(p)
            self.assertFalse(p["outcome_supervision_mask"])
            self.assertFalse(p["task_complete"])
            self.assertNotIn("memory_prefix", p["memory"])

    def test_overflow_rejected_not_shifted_or_stretched(self):
        annotation = dict(meta_data=dict(valid_duration=[180, 400]), skill_annotation=[raw_skill(0, 180, 400)])
        with self.assertRaises(ValueError):
            compile_episode(annotation, dict(length=220), "test")

    def test_unannotated_ends_are_not_success(self):
        annotation = dict(meta_data=dict(valid_duration=[10, 100]), skill_annotation=[raw_skill(0, 10, 80)])
        segments = compile_episode(annotation, dict(length=280), "test")
        self.assertEqual([(s["start"], s["end"]) for s in segments], [(10, 80)])
        self.assertTrue(all(a[0] < 80 for a in anchor_records(segments, 0)))

    def test_phase_ignores_dataset_reindex(self):
        a = dict(task_index=4, raw_episode_id=40010, task_instance_id=1, episode_index=100)
        self.assertEqual(fixed_phase(a), fixed_phase(dict(a, episode_index=9000)))


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('periodic_teacher',
    Path(__file__).resolve().parents[1] / 'scripts/rl/memlite_online/code/recovery_periodic_teacher.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PeriodicTeacherTests(unittest.TestCase):
    def trace(self):
        segments = [dict(start=0, end=64, text='NAVIGATE', parent='P0'),
                    dict(start=64, end=192, text='GRASP', parent='P1'),
                    dict(start=192, end=512, text='PLACE', parent='P2')]
        rows = []
        previous = parent = 'None'
        for frame in range(0, 512, 16):
            index = next(i for i, s in enumerate(segments) if s['start'] <= frame < s['end'])
            rows.append((frame, index, previous, parent, []))
            previous, parent = segments[index]['text'], segments[index]['parent']
        return rows, segments

    def test_previous_intent_and_memory_recomputed_not_only_refresh_counter(self):
        anchors, segments = self.trace()
        records, audit = module.periodic_annotation_trace(anchors, segments)
        self.assertEqual([r['frame'] for r in records], [0, 128, 256, 384])
        self.assertEqual(anchors[8][2], 'GRASP')
        self.assertEqual(records[1]['previous_intent'], 'NAVIGATE')
        self.assertEqual(records[2]['previous_intent'], 'GRASP')
        self.assertEqual(records[2]['history'], ['NAVIGATE'])
        self.assertEqual(module.command_kind(records[1], segments[records[1]['segment']]), 'transition')
        self.assertEqual(module.command_kind(records[-1], segments[records[-1]['segment']]), 'continuation')
        self.assertIsNone(audit['first_unannotated_scheduled_call'])

    def test_annotation_gap_never_invents_previous_command(self):
        anchors, segments = self.trace()
        anchors = [row for row in anchors if row[0] != 128]
        records, audit = module.periodic_annotation_trace(anchors, segments)
        self.assertEqual(len(records), 1)
        self.assertEqual(audit['first_unannotated_scheduled_call'], 128)
        self.assertEqual(audit['omitted_known_calls_after_gap'], 2)

    def test_noncausal_clock_and_arbitrary_period_rejected(self):
        anchors, segments = self.trace()
        with self.assertRaises(ValueError): module.periodic_annotation_trace(anchors[::-1], segments)
        with self.assertRaises(ValueError): module.periodic_annotation_trace(anchors, segments, period=16)


if __name__ == '__main__': unittest.main()

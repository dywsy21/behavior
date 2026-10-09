from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sign_local_recovery_reviews import verify_action_window


class LocalReviewSigningTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(label_kind='injected_fault_not_BC' if i<32 else 'same_state_local_teacher_candidate')
                   for i in range(96)]
        self.manifest=dict(anchors={str(t):{} for t in range(0,96,4)})
        self.anchor=dict(label_audit=dict(full_executed_32_step_target_available=True))
        self.frames=list(range(32,65,4))

    def test_exact_reviewed_clean_window(self):
        verify_action_window(32,self.rows,self.manifest,self.anchor,self.frames)

    def test_injected_or_unexecuted_targets_rejected(self):
        with self.assertRaises(ValueError):verify_action_window(0,self.rows,self.manifest,self.anchor,list(range(0,33,4)))
        with self.assertRaises(ValueError):verify_action_window(32,self.rows[:63],self.manifest,self.anchor,self.frames)
        row=deepcopy(self.anchor);row['label_audit']['full_executed_32_step_target_available']=False
        with self.assertRaises(ValueError):verify_action_window(32,self.rows,self.manifest,row,self.frames)

    def test_missing_interior_or_endpoint_review_rejected(self):
        for missing in (32,44,64):
            frames=[t for t in self.frames if t!=missing]
            with self.assertRaises(ValueError):verify_action_window(32,self.rows,self.manifest,self.anchor,frames)


if __name__=='__main__':unittest.main()

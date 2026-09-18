"""Offline evidence audit: late grasps and missing receipts must not be missed."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[2]/"scripts/semantic_robot/review_saved_run.py"
spec=importlib.util.spec_from_file_location("saved_run_review",path)
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def row(i,move,obj,receipt=True):
    return {"decision":i,"execution":{"action":{"part":"right","move":move},"feedback":{}},
            "post_action_grasp_audit":{"assisted_objects":{"left":None,"right":obj}} if receipt else None}


class SavedGraspAuditTests(unittest.TestCase):
    def test_late_normal_close_is_not_lost_after_empty_probes(self):
        rows=[row(5,"close",None),row(6,"open",None),row(9,"close",None),row(10,"open",None),
              row(21,"close","radio_89"),row(22,"up","radio_89"),row(23,"up","radio_89"),
              row(24,"up","radio_89"),row(25,"open",None)]
        r=review.grasp_timeline(rows)
        self.assertEqual(r["close_decisions"],[5,9,21])
        self.assertEqual([x["decision"] for x in r["attachment_observations"]],[21,22,23,24])
        self.assertEqual(r["opens_after_observed_attachment"][0]["decision"],25)
        self.assertEqual(r["opens_after_observed_attachment"][0]["object_before"],"radio_89")
        self.assertTrue(r["not_task_success"])

    def test_missing_receipt_is_unknown_not_empty_or_continuing_attachment(self):
        rows=[row(1,"close","radio_89"),row(2,"up",None,False),row(3,"open",None)]
        r=review.grasp_timeline(rows)
        self.assertEqual(r["missing_post_action_receipts"],[2])
        self.assertEqual(r["opens_after_observed_attachment"],[])
        self.assertEqual(r["attachment_changes"],[])

    def test_unexecuted_observation_and_registration_are_separate(self):
        r1=row(2,"up","radio_89");r1["registered_grasp_motion"]={"verified":True}
        r2=row(3,"open",None);r2["execution"]=None
        r=review.grasp_timeline([r1,r2])
        self.assertEqual(r["registered_verification_decisions"],[2])
        self.assertEqual(r["open_decisions"],[])
        self.assertEqual(len(r["attachment_observations"]),1)

    def test_physical_mode_has_no_assisted_diagnostic(self):
        x=row(0,"close",None);x["post_action_grasp_audit"]["assisted_objects"]=None
        r=review.grasp_timeline([x])
        self.assertEqual(r["unavailable_attachment_diagnostics"],[0])
        self.assertEqual(r["attachment_observations"],[])


if __name__=="__main__":unittest.main()

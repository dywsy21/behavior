from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/vlm_sft"))
from summarize_local import rotation_path
from stratify_static import intent_coverage


class OfflineSummaryTests(unittest.TestCase):
    def test_exact_intent_counts_do_not_alias_same_task_or_other_task(self):
        row={"task_id":0,"active_instruction":"GRASP radio","proprio":{"base_velocity_local":[0,0,0]},
            "history":[],"episode_index":12,"target":"RIGHT_CLOSE"}
        rows=[row,{**row,"active_instruction":"NAVIGATE radio"},{**row,"task_id":3}]
        result=intent_coverage(rows,0,"GRASP radio")
        self.assertEqual(result["task_rows"],2)
        self.assertEqual(result["exact_intent_rows"],1)
        self.assertEqual(result["low_base_velocity_rows"],1)
        self.assertEqual(result["source_episodes"],[12])
        self.assertEqual(intent_coverage(rows,3,"NAVIGATE table")["exact_intent_rows"],0)

    def test_full_turn_is_not_confused_with_wrapped_endpoint(self):
        quaternions=Rotation.from_euler("z",[0,-90,-180,-270,-360,-400],degrees=True).as_quat()
        result=rotation_path(quaternions)
        self.assertAlmostEqual(np.rad2deg(result["decision_endpoint_signed_yaw_travel_rad"]),-400)
        self.assertAlmostEqual(np.rad2deg(result["decision_endpoint_rotation_travel_rad"]),400)

    def test_return_to_start_has_path_but_zero_net_heading(self):
        quaternions=Rotation.from_euler("z",[0,30,0],degrees=True).as_quat()
        result=rotation_path(quaternions)
        self.assertAlmostEqual(result["decision_endpoint_signed_yaw_travel_rad"],0)
        self.assertAlmostEqual(np.rad2deg(result["decision_endpoint_absolute_yaw_travel_rad"]),60)


if __name__=="__main__":unittest.main()

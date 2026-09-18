from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/semantic_robot"))
from audit_inspection_view import angle,view_direction


class InspectionViewAuditTests(unittest.TestCase):
    def fixture(self):
        hand=np.eye(4);head=np.eye(4);head[2,3]=1.
        return head,hand,np.zeros(3)

    def test_radial_translation_changes_pose_not_side(self):
        head,hand,point=self.fixture();first=view_direction(head,hand,point)
        hand[2,3]=.2
        self.assertAlmostEqual(angle(first,view_direction(head,hand,point)),0.)

    def test_roll_about_view_axis_does_not_reveal_new_side(self):
        head,hand,point=self.fixture();first=view_direction(head,hand,point)
        hand[:3,:3]=Rotation.from_euler("z",30,degrees=True).as_matrix()
        self.assertAlmostEqual(angle(first,view_direction(head,hand,point)),0.)

    def test_transverse_rotation_changes_relative_view(self):
        head,hand,point=self.fixture();first=view_direction(head,hand,point)
        hand[:3,:3]=Rotation.from_euler("x",30,degrees=True).as_matrix()
        self.assertAlmostEqual(angle(first,view_direction(head,hand,point)),30.)

    def test_common_rigid_motion_is_not_new_relative_view(self):
        head,hand,point=self.fixture();first=view_direction(head,hand,point)
        world=np.eye(4);world[:3,:3]=Rotation.from_euler("xyz",[21,33,40],degrees=True).as_matrix();world[:3,3]=[2,3,4]
        np.testing.assert_allclose(first,view_direction(world@head,world@hand,point),atol=1e-12)

    def test_coincident_camera_anchor_rejected(self):
        with self.assertRaises(ValueError):view_direction(np.eye(4),np.eye(4),np.zeros(3))


if __name__=="__main__":unittest.main()

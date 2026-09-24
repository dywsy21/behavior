import copy
import unittest
import numpy as np

from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.proposal_geometry import box_samples,diagnose_box
from test_v2 import fixture


def inputs(mesh=True):
    model,state=fixture();spec=copy.deepcopy(model.spec)
    spec['links']['link:base_link']={'T_reference':np.eye(4).tolist(),'screws':np.zeros((6,18)).tolist()}
    if mesh:
        spec['metadata']['base_visual_surface']={'source':'robot_base_visual_mesh_only','scene_truth':False,
            'link':'link:base_link','vertices':[[-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]],'faces':[[0,1,2],[0,2,3]]}
    model=RobotModel(spec)
    raw={v:np.zeros((100,100,3),np.uint8) for v in spec['metadata']['cameras']}
    depths={v:np.ones((100,100),np.float32) for v in raw}
    return dict(frame_id='frozen-test-1',goal=Goal('pick','an object','right','independent result'),
                raw=raw,depths=depths,model=model,state=state,view='head',box=[10.,10.,90.,90.])


class ProposalGeometryTests(unittest.TestCase):
    def test_uniform_bounded_sampling_and_original_box_unchanged(self):
        box=[-10.,1.,120.,99.];before=box.copy();rows=box_samples(box,100,100)
        self.assertEqual(box,before);self.assertEqual(rows.shape,(144,2))
        self.assertTrue(np.all((rows>=0)&(rows<100)))
        self.assertEqual(len(set(rows[:,0])),12);self.assertEqual(len(set(rows[:,1])),12)
        self.assertEqual(len(box_samples([.01,.01,.05,.05],100,100)),1)
        self.assertEqual(len(box_samples([110,0,130,10],100,100)),0)
        for bad in ([0,0,0,1],[1,1,0,0],[0,0,float('nan'),1],[False,0,1,1],None):
            with self.subTest(bad=bad),self.assertRaises(ValueError):box_samples(bad,100,100)

    def test_half_open_pixel_cells_include_last_column_and_never_round_outward(self):
        np.testing.assert_array_equal(box_samples([99,0,100,1],100,100),[[99,0]])
        np.testing.assert_array_equal(box_samples([0,99,1,100],100,100),[[0,99]])
        np.testing.assert_array_equal(box_samples([99,99,100,100],100,100),[[99,99]])
        np.testing.assert_array_equal(box_samples([10,10,11,11],100,100),[[10,10]])
        np.testing.assert_array_equal(box_samples([-1,-1,.4,.4],100,100),[[0,0]])
        self.assertEqual(len(box_samples([100,0,101,1],100,100)),0)
        self.assertEqual(len(box_samples([0,100,1,101],100,100)),0)

    def test_exact_chassis_surface_not_all_points_in_robot_envelope(self):
        args=inputs();result=diagnose_box(**args)
        self.assertEqual(result['chassis_samples'],144)
        self.assertEqual(result['chassis_fraction_of_depth_samples'],1.)
        self.assertTrue(result['diagnostic_only']);self.assertTrue(result['not_complete_robot_mask'])
        self.assertIsNone(result['target_identity']);self.assertIsNone(result['target_accepted'])
        args['depths']['head'][:]=1.1
        outside=diagnose_box(**args)
        self.assertEqual(outside['chassis_samples'],0)
        self.assertIsNone(outside['target_accepted'])
        self.assertNotEqual(result['frame_binding'],outside['frame_binding'])

    def test_missing_mesh_is_unknown_not_unmasked_target(self):
        result=diagnose_box(**inputs(mesh=False))
        self.assertFalse(result['chassis_geometry']['available'])
        self.assertIsNone(result['chassis_samples']);self.assertIsNone(result['chassis_fraction_of_depth_samples'])
        self.assertTrue(all(row['on_chassis_surface'] is None for row in result['samples']))

    def test_invalid_depth_is_not_repaired_and_outside_box_is_empty(self):
        args=inputs();args['depths']['head'][:]=np.nan
        result=diagnose_box(**args)
        self.assertEqual(result['depth_valid_samples'],0);self.assertEqual(result['invalid_depth_samples'],144)
        self.assertIsNone(result['chassis_fraction_of_depth_samples'])
        args['box']=[110,0,130,20]
        self.assertEqual(diagnose_box(**args)['sampled_pixels'],0)

    def test_changed_calibration_arrays_and_nonrobot_mesh_fail_closed(self):
        args=inputs();args['model'].reference[0]+=.1
        with self.assertRaises(ValueError):diagnose_box(**args)
        args=inputs();spec=copy.deepcopy(args['model'].spec)
        spec['metadata']['base_visual_surface']['scene_truth']=True
        args['model']=RobotModel(spec)
        with self.assertRaises(ValueError):diagnose_box(**args)

    def test_camera_geometry_cannot_be_singular_nonfinite_or_nonrigid(self):
        for mode in ('zero_focal','nan','nonrigid'):
            args=inputs();spec=copy.deepcopy(args['model'].spec)
            if mode=='zero_focal':spec['metadata']['cameras']['head']['K'][0][0]=0
            elif mode=='nan':spec['metadata']['cameras']['head']['K'][1][1]=float('nan')
            else:spec['links']['camera_head']['T_reference'][0][0]=2
            args['model']=RobotModel(spec)
            with self.subTest(mode=mode),self.assertRaises(ValueError):diagnose_box(**args)


if __name__=='__main__':unittest.main()

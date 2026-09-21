import unittest
from unittest.mock import patch

import cv2
import numpy as np

from semantic_robot.v2.feature_tracking import refine_matches, VERSION
from semantic_robot.v2.odometry import RGBDMotion


class FeatureTrackingTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(31)
        image=rng.integers(0,256,(120,160),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),.6)
        points=np.array([[x,y] for x in (35.,60.,85.,110.,135.) for y in (35.,60.,85.)])
        return image,points

    def test_known_subpixel_translation_corrects_initial_location(self):
        before,points=self.data();delta=np.array([1.5,-.75])
        after=cv2.warpAffine(before,np.array([[1.,0.,delta[0]],[0.,1.,delta[1]]]),(160,120))
        guess=points+delta+[.55,-.35];original=guess.copy()
        result,keep,receipt=refine_matches(before,after,points,guess)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(result-points,np.tile(delta,(len(points),1)),atol=.16)
        np.testing.assert_array_equal(guess,original)
        self.assertEqual(receipt['version'],VERSION)
        self.assertFalse(receipt['unrefined_fallback'])

    def test_empty_has_no_opencv_call_and_no_fictitious_matches(self):
        image,_=self.data()
        with patch('cv2.calcOpticalFlowPyrLK') as tracking:
            result,keep,receipt=refine_matches(image,image,np.empty((0,2)),np.empty((0,2)))
        tracking.assert_not_called();self.assertEqual(result.shape,(0,2));self.assertEqual(keep.size,0)
        self.assertEqual(receipt['bidirectional_accepted'],0)

    def test_invalid_shape_dtype_nonfinite_are_rejected(self):
        image,points=self.data()
        variants=[(image.astype(float),image,points,points),(image,image[:100],points,points),
                  (image[:,:,None],image[:,:,None],points,points),(image,image,points[:,0],points[:,0]),
                  (image,image,points,points[:-1]),(image,image,points+np.nan,points)]
        for args in variants:
            with self.subTest(shape=[x.shape for x in args]),self.assertRaises(ValueError):refine_matches(*args)

    def test_border_and_off_image_coordinates_are_not_tracked(self):
        image,_=self.data();points=np.array([[0.,10.],[159.,10.],[12.,120.],[-1.,10.]])
        with patch('cv2.calcOpticalFlowPyrLK') as tracking:
            _,keep,_=refine_matches(image,image,points,points)
        tracking.assert_not_called();self.assertFalse(keep.any())

    def test_forward_failure_nonfinite_and_outside_do_not_reach_backward(self):
        image,points=self.data();n=len(points)
        for out,status in [(points[:,None].astype(np.float32),np.zeros((n,1),np.uint8)),
                           (np.full((n,1,2),np.nan,np.float32),np.ones((n,1),np.uint8)),
                           (np.full((n,1,2),9999.,np.float32),np.ones((n,1),np.uint8))]:
            with patch('cv2.calcOpticalFlowPyrLK',return_value=(out,status,None)) as tracking:
                _,keep,_=refine_matches(image,image,points,points)
            self.assertEqual(tracking.call_count,1);self.assertFalse(keep.any())

    def test_reverse_inconsistency_and_failure_rejected_without_fallback(self):
        image,points=self.data();n=len(points);good=np.ones((n,1),np.uint8)
        for reverse,status in [(points[:,None]+[2.,0.],good),(points[:,None],good*0),
                               (points[:,None]*np.nan,good)]:
            with patch('cv2.calcOpticalFlowPyrLK',side_effect=[(points[:,None],good,None),(reverse,status,None)]):
                _,keep,receipt=refine_matches(image,image,points,points)
            self.assertFalse(keep.any());self.assertFalse(receipt['unrefined_fallback'])

    def test_error_or_none_returns_zero_valid_not_original_support(self):
        image,points=self.data()
        for side in [cv2.error('fixture'),[(None,None,None)]]:
            with patch('cv2.calcOpticalFlowPyrLK',side_effect=side):
                _,keep,receipt=refine_matches(image,image,points,points)
            self.assertFalse(keep.any());self.assertFalse(receipt['unrefined_fallback'])

    def test_malformed_tracker_shape_fails_closed(self):
        image,points=self.data()
        with patch('cv2.calcOpticalFlowPyrLK',return_value=(points[:1,None],np.ones((1,1)),None)):
            with self.assertRaises(ValueError):refine_matches(image,image,points,points)

    def test_explicit_joint_only_and_default_is_unchanged(self):
        self.assertFalse(RGBDMotion().refine_matches)
        self.assertTrue(RGBDMotion('rgbd_joint',refine_matches=True).refine_matches)
        for args in [dict(refine_matches=1),dict(refine_matches=None),dict(refine_matches=True),
                     dict(estimator='rgbd_rigid',refine_matches=True)]:
            with self.assertRaises(ValueError):RGBDMotion(**args)


if __name__=='__main__':unittest.main()

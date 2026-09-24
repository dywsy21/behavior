import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

import replay_target_self_veto as replay
from semantic_robot.v2.finite_localization import bind_frame
from semantic_robot.v2.grounding import unproject
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.native_grounding import native_request
from test_target_self_veto import setup_surface


def inputs():
    model,state,depths=setup_surface()
    return dict(frame_id='test-public-frame',goal=Goal('pick','object','right','independent result'),
        raw={v:np.zeros((100,100,3),np.uint8) for v in depths},depths=depths,model=model,state=state)


def rows(arguments,self_point=True):
    binding=bind_frame(**arguments)
    detection={'box_xyxy_px':[50.,50.,51.,51.],'score':.5,'text_label':'object'}
    query={'id':'synthetic__head','view':'head','frame_binding':binding,'detections':[detection]}
    model=arguments['model'];camera=model.spec['metadata']['cameras']['head']
    point=unproject([[50,50]],[arguments['depths']['head'][50,50]],camera['K'],
                    model.forward(arguments['state'].q,'camera_head'))[0]
    diagnostic={**copy.deepcopy(query),'geometry':[{'frame_binding':binding,'view':'head',
        'box_xyxy_px':detection['box_xyxy_px'].copy(),
        'samples':[{'pixel_xy':[50,50],'point_base_m':point.tolist(),'on_chassis_surface':self_point}]}]}
    return query,diagnostic


class SavedSelfReplayTests(unittest.TestCase):
    def test_original_self_and_nonself_samples_remain_distinct_without_semantic_labels(self):
        args=inputs();query,diag=rows(args)
        result=replay.replay_pixels(args,query,diag,time.monotonic()+10)
        self.assertEqual((result['total'],result['original_self'],result['valid'],result['self_still_valid']),(1,1,0,0))
        self.assertEqual(result['reason_counts'],{'TARGET_ON_ROBOT_CHASSIS':1})
        for depth in args['depths'].values():depth[:]=1.1
        query,diag=rows(args,False)
        result=replay.replay_pixels(args,query,diag,time.monotonic()+10)
        self.assertEqual((result['original_self'],result['valid']),(0,1))
        self.assertTrue(result['samples_are_not_model_predictions_or_positive_semantic_labels'])

    def test_no_detection_is_kept_as_empty_not_fabricated_point(self):
        args=inputs();query,diag=rows(args);query['detections']=[];diag['detections']=[];diag['geometry']=[]
        result=replay.replay_pixels(args,query,diag,time.monotonic()+10)
        self.assertEqual(result['total'],0);self.assertEqual(result['samples'],[])

    def test_wrong_frame_box_pixel_set_depth_or_unknown_self_is_rejected(self):
        for mutation in ('frame','box','pixel','depth','self'):
            args=inputs();query,diag=rows(args);box=diag['geometry'][0]
            if mutation=='frame':box['frame_binding']='old-frame'
            elif mutation=='box':box['box_xyxy_px'][0]=49.
            elif mutation=='pixel':box['samples']=[]
            elif mutation=='depth':box['samples'][0]['point_base_m'][0]+=.01
            else:box['samples'][0]['on_chassis_surface']=None
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                replay.replay_pixels(args,query,diag,time.monotonic()+10)

    def test_expired_deadline_stops_before_first_contact(self):
        args=inputs();query,diag=rows(args)
        with self.assertRaises(TimeoutError):replay.replay_pixels(args,query,diag,time.monotonic()-1)

    def test_native_answer_is_replayed_only_for_identical_request(self):
        args=inputs();binding=bind_frame(**args)
        request=native_request(args['goal'],'head',args['raw']['head'],binding,'point')
        old={'binding':binding,'native_point':{'evidence':None,'receipt':{
            'binding':binding,'view':'head','request_sha256':request.sha256,'request':request.receipt(),
            'raw_text':'[{"point_2d":[500,500],"label":"object"}]'}}}
        result=replay.replay_native(args,old,time.monotonic()+10)
        self.assertEqual(result['actual_model_calls'],0);self.assertEqual(result['historical_model_answer_count'],1)
        self.assertFalse(result['new_evidence']['visible'])
        self.assertEqual(result['new_geometry']['reason'],'TARGET_ON_ROBOT_CHASSIS')
        old['native_point']['receipt']['request_sha256']='wrong'
        with self.assertRaises(ValueError):replay.replay_native(args,old,time.monotonic()+10)

    def test_frozen_result_bytes_must_match_exact_sha_and_not_symlink(self):
        data=b'{"status":"complete"}';digest=hashlib.sha256(data).hexdigest()
        with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'read_bytes',return_value=data):
            self.assertEqual(replay.frozen_json(Path('unused'),digest),{'status':'complete'})
            with self.assertRaises(ValueError):replay.frozen_json(Path('unused'),'bad')
        with patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaises(ValueError):replay.frozen_json(Path('unused'),digest)


if __name__=='__main__':unittest.main()

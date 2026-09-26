from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pyarrow as pa

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/vlm_sft'))
import prepare_expert_action_corpus as corpus


class CorpusTests(unittest.TestCase):
    def test_identity_cannot_mix_frozen_cohort_with_new_labels(self):
        identity={'labels_sha256':'fixed-labels','quarantine_sha256':'fixed-quarantine','episode_meta_sha256':{'meta':'fixed-meta'}}
        content=json.dumps({'source_identity':identity}).encode()
        raw={'source_identity':{**identity,'counts_sha256':hashlib.sha256(content).hexdigest()}}
        self.assertEqual(corpus.bound_identity(raw,content),identity)
        changed=json.dumps({'source_identity':{**identity,'labels_sha256':'new-labels'}}).encode()
        with self.assertRaises(ValueError):corpus.bound_identity(raw,changed)

    def test_video_time_bounds_reject_short_or_shifted_stream(self):
        corpus.validate_video_window(10.,30,0.,11.)
        with self.assertRaises(ValueError):corpus.validate_video_window(10.,30,0.,10.8)
        with self.assertRaises(ValueError):corpus.validate_video_window(0.,30,1.,10.)

    def sources(self):
        rows=[]
        for task in range(5):
            for instance in range(96):
                rows.append({'task':task,'instance':instance,'episode':task*200+instance,
                    'frames':50,'split':'train' if instance<80 else 'validation' if instance<88 else 'test'})
        review=[{'protect_groups_in_future_student_releases':[[0,i] for i in range(14)]}]
        return {'sources':rows},review

    def test_frozen_grouping_and_calibration_exclusion(self):
        plan,review=self.sources();chosen=corpus.select_corpus_sources(plan,review)
        self.assertEqual(len(chosen),466)
        self.assertTrue(all((s['task'],s['instance']) not in {(0,i) for i in range(14)} for s in chosen))
        plan['sources'].append(plan['sources'][-1])
        with self.assertRaises(ValueError):corpus.select_corpus_sources(plan,review)

    def fixture(self):
        n=50;source={'task':0,'instance':25,'episode':10,'frames':n,'split':'train'}
        states=np.arange(n*61,dtype=np.float32).reshape(n,61)/100
        actions=np.sin(np.arange(n*23,dtype=np.float32).reshape(n,23))*.8
        labels=[{'frame_index':f,'memlite_branch':'low','source_kind':'original_demo',
            'low_action_supervision_mask':True,'action_horizon_end':n,'segment_end':n,
            'active_skills_semantic_json':'[{"verb":"GRASP","target":"radio","arm":"right"}]'} for f in range(n)]
        return source,states,actions,labels

    def test_exact_current_state_and_full_action_chunk(self):
        source,states,actions,labels=self.fixture()
        rows,_=corpus.make_rows(source,states,actions,labels,[],'Turn on the radio')
        self.assertEqual([r['frame_index'] for r in rows],[0,16,32])
        for row in rows:
            f=row['frame_index']
            np.testing.assert_array_equal(row['observation_state'],states[f])
            np.testing.assert_array_equal(row['expert_action'],actions[f:f+16])
            self.assertEqual(set(row),{'id','source_split','task_index','task_instance_id','episode_index',
                'frame_index','timestamp_s','task','active_instruction','observation_state','expert_action'})
        corpus.verify_payload(pa.Table.from_pylist(rows),rows,states,actions)
        for dimension in (0,3,6,7,14,15,22):
            bad=deepcopy(rows);bad[1]['expert_action'][0][dimension]+=.01
            with self.assertRaises(ValueError):corpus.verify_payload(pa.Table.from_pylist(bad),rows,states,actions)

    def test_ambiguous_labels_and_cross_quarantine_rejected(self):
        source,states,actions,labels=self.fixture()
        rows,rejected=corpus.make_rows(source,states,actions,labels,[(16,16)],'Task')
        self.assertEqual([r['frame_index'] for r in rows],[32])
        self.assertEqual(rejected['not_released_same_skill'],2)
        with self.assertRaises(ValueError):corpus.make_rows(source,states,actions,labels+[labels[0]],[],'Task')

    def test_source_clock_and_all_controls_checked(self):
        source,states,actions,_=self.fixture()
        original={'frame_index':list(range(50)),'timestamp':np.arange(50)/30,
                  'observation.state':states.tolist(),'action':actions.tolist()}
        got=corpus.arrays_from_source(pa.table(original),source)
        np.testing.assert_array_equal(got[1],actions)
        bad=deepcopy(original);bad['timestamp']=np.arange(50)/30+.0333
        with self.assertRaises(ValueError):corpus.arrays_from_source(pa.table(bad),source)
        bad=deepcopy(original);bad['action'][3][22]=1.5
        with self.assertRaises(ValueError):corpus.arrays_from_source(pa.table(bad),source)
        bad=deepcopy(original);bad['observation.state'][5][0]=float('nan')
        with self.assertRaises(ValueError):corpus.arrays_from_source(pa.table(bad),source)


if __name__=='__main__':unittest.main()

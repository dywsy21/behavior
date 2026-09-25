from copy import deepcopy
from contextlib import contextmanager, ExitStack
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/vlm_sft'))
from prepare_visual_review import (select_sources,frames_for,choose_frame,COUNTS,COUNTS_SHA,
                                   LATER_PROTECTED,PRIOR_STATE_TRAINING,QUERIES)
import prepare_visual_review as visual


class VisualReviewTests(unittest.TestCase):
    @contextmanager
    def preparation_fixture(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);meta_path=root/'episodes.parquet';meta_path.write_bytes(b'metadata')
            quarantine=root/'quarantine.parquet';quarantine.write_bytes(b'quarantine')
            counts=json.loads(COUNTS.read_text())
            counts['source_identity']['episode_meta_sha256']={str(meta_path):visual.sha(meta_path)}
            counts['source_identity']['quarantine_sha256']=visual.sha(quarantine)
            count_path=root/'counts.json';count_path.write_text(json.dumps(counts))
            meta=[]
            for source in select_sources(counts):
                row={'episode_index':source['episode'],'task_index':source['task'],
                     'task_instance_id':source['instance'],'length':source['frames']}
                for camera in visual.CAMERAS.values():
                    stem='videos/observation.rgb.'+camera
                    row.update({stem+'/chunk_index':0,stem+'/file_index':source['episode'],stem+'/from_timestamp':0.})
                    video=root/f'{stem}/chunk-000/file-{source["episode"]:03d}.mp4'
                    video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(b'fake_video_for_mock_decoder')
                meta.append(row)
            def read_table(path):return SimpleNamespace(to_pylist=lambda:[] if Path(path)==quarantine else meta)
            pq=SimpleNamespace(read_table=read_table)
            class Decoder:
                def __init__(self):
                    self.streams=SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1,30),average_rate=30,
                        codec_context=SimpleNamespace(thread_count=0))]);self.position=0
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def seek(self,position,**kwargs):self.position=position
                def decode(self,stream):
                    return iter(SimpleNamespace(pts=p,to_image=lambda:Image.new('RGB',(8,8),'red'))
                                for p in range(self.position-1,self.position+2))
            av=SimpleNamespace(open=Mock(side_effect=lambda path:Decoder()))
            stack.enter_context(patch.dict(sys.modules,{'av':av,'pyarrow':SimpleNamespace(parquet=pq),'pyarrow.parquet':pq}))
            for key,value in {'COUNTS':count_path,'COUNTS_SHA':visual.sha(count_path),'QUARANTINE':quarantine,'ROOT':root}.items():
                stack.enter_context(patch.object(visual,key,value))
            stack.enter_context(patch.object(visual.os,'sched_setaffinity'))
            stack.enter_context(patch.object(visual.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)))
            stack.enter_context(patch.object(visual.subprocess,'check_output',side_effect=lambda args,**kw:'' if 'status' in args else 'a'*40))
            stack.enter_context(patch('builtins.print'))
            yield root/'output',av

    def test_real_prepare_seals_exact_unreleased_collection_and_tamper_is_rejected(self):
        with self.preparation_fixture() as (out,av):
            result=visual.prepare(out)
            self.assertEqual(visual.validate_collection(out),result)
            self.assertEqual((len(result['rows']),result['image_count']),(36,108))
            self.assertEqual(av.open.call_count,36)
            self.assertTrue(all(r['training_eligible'] is False and len(r['images'])==3 for r in result['rows']))
            first=out/result['rows'][0]['images']['head'];first.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'content changed'):visual.validate_collection(out)

    def test_real_prepare_partial_decode_never_becomes_complete_and_primary_survives_record_error(self):
        for failure_record in (False,True):
            with self.preparation_fixture() as (out,av):
                primary=ValueError('DECODE_PRIMARY');av.open.side_effect=primary
                writer=visual.atomic_json
                def failing_writer(path,value):
                    if failure_record and path.name=='failure.json':raise OSError('RECORD_SECONDARY')
                    return writer(path,value)
                with patch.object(visual,'atomic_json',side_effect=failing_writer),self.assertRaises(ValueError) as caught:
                    visual.prepare(out)
                self.assertIs(caught.exception,primary)
                self.assertFalse((out/'complete.json').exists())
                with self.assertRaises((ValueError,FileNotFoundError)):visual.validate_collection(out)

    def test_missing_or_partial_seal_is_not_completed_even_with_all_images(self):
        with self.preparation_fixture() as (out,_):
            visual.prepare(out)
            (out/'complete.json').write_text('{')
            with self.assertRaises(ValueError):visual.validate_collection(out)

    def test_resealed_group_counter_budget_or_missing_start_cannot_bypass_validation(self):
        with self.preparation_fixture() as (out,_):
            visual.prepare(out)
            manifest=out/'review_manifest.json';seal_path=out/'complete.json'
            original=json.loads(manifest.read_text());seal=json.loads(seal_path.read_text())
            variants=[lambda d:d.update(wall_seconds=999),lambda d:d.update(new_model_calls=7),
                lambda d:d.update(training_updates=True),lambda d:d.update(image_bytes=1024**3+1),
                lambda d:d['rows'][0].update(task=d['rows'][3]['task'],instance=d['rows'][3]['instance']),
                lambda d:d['rows'][0].update(frame=1),lambda d:d['sources'][0].update(instance=999)]
            for mutate in variants:
                changed=deepcopy(original);mutate(changed);manifest.write_text(json.dumps(changed))
                seal['manifest_sha256']=visual.sha(manifest);seal_path.write_text(json.dumps(seal))
                with self.assertRaises(ValueError):visual.validate_collection(out)
            manifest.write_text(json.dumps(original));seal['manifest_sha256']=visual.sha(manifest)
            seal_path.write_text(json.dumps(seal));(out/'extraction_started.json').rename(out/'saved_start.json')
            with self.assertRaises(FileNotFoundError):visual.validate_collection(out)

    def test_real_prepare_budget_failure_even_after_seal_invalidates_collection(self):
        for phase in ('start','after_seal'):
            with self.preparation_fixture() as (out,_):
                late={'value':False,'calls':0};writer=visual.atomic_json
                def now():
                    late['calls']+=1
                    return 241. if late['value'] or (phase=='start' and late['calls']>1) else 0.
                def save(path,value):
                    writer(path,value)
                    if path.name=='complete.json':late['value']=True
                with patch.object(visual.time,'monotonic',side_effect=now),patch.object(visual,'atomic_json',side_effect=save):
                    with self.assertRaises(TimeoutError):visual.prepare(out)
                self.assertTrue((out/'failure.json').exists())
                with self.assertRaisesRegex(ValueError,'partial'):visual.validate_collection(out)

    def counts(self):
        return {'exclusions':{'old':[[0,15]]},'sources':[
            {'cohort':'additional_train','task':task,'instance':i,'episode':task*200+i,'frames':100}
            for task in QUERIES for i in (1,11,12,13,14,15,71,114,138,192,242)]}

    def test_split_is_deterministic_by_source_instance_before_images(self):
        counts=self.counts();first=select_sources(counts)
        counts['sources'].reverse();self.assertEqual(first,select_sources(counts))
        self.assertEqual(len(first),12)
        groups={(r['task'],r['instance']) for r in first}
        self.assertEqual(len(groups),12)
        self.assertFalse(groups & (LATER_PROTECTED|PRIOR_STATE_TRAINING|{(0,15)}))
        self.assertEqual(sum(r['candidate_split']=='visual_validation' for r in first),3)

    def test_original_train_cohort_duplicate_or_insufficient_candidates_cannot_slip_in(self):
        for mode in ('old_cohort','duplicate','short'):
            counts=self.counts()
            if mode=='old_cohort':
                for row in counts['sources']:row['cohort']='h09_train'
            if mode=='duplicate':counts['sources'].append(deepcopy(counts['sources'][1]))
            if mode=='short':counts['sources']=counts['sources'][:2]
            with self.subTest(mode=mode),self.assertRaises(ValueError):select_sources(counts)

    def test_frozen_real_manifest_has_twelve_allowed_groups_and_disjoint_candidate_splits(self):
        raw=COUNTS.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),COUNTS_SHA)
        rows=select_sources(json.loads(raw))
        groups={split:{(r['task'],r['instance']) for r in rows if r['candidate_split']==split}
                for split in ('visual_train','visual_validation')}
        self.assertEqual(tuple(map(len,groups.values())),(9,3))
        self.assertFalse(groups['visual_train'] & groups['visual_validation'])
        self.assertFalse((groups['visual_train']|groups['visual_validation']) & PRIOR_STATE_TRAINING)

    def test_fixed_frames_quarantine_and_no_replacement(self):
        source={'frames':101,'episode':3}
        self.assertEqual(frames_for(source,[]),[0,50,90])
        for frame in (0,50,90):
            with self.assertRaises(ValueError):frames_for(source,[{'episode_index':3,'frame_start':frame,'frame_end':frame}])
        self.assertEqual(frames_for(source,[{'episode_index':4,'frame_start':0,'frame_end':100}]),[0,50,90])

    def test_native_pts_not_seek_request_is_the_frame_receipt(self):
        stream=SimpleNamespace(time_base=Fraction(1,30))
        container=Mock();container.decode.return_value=iter([SimpleNamespace(pts=None),SimpleNamespace(pts=29),SimpleNamespace(pts=30)])
        frame,actual=choose_frame(container,stream,1.)
        self.assertEqual(frame.pts,30);self.assertEqual(actual,1.)
        container.seek.assert_called_once_with(30,stream=stream,backward=True)
        for pts in ([],[31]):
            container.decode.return_value=iter(SimpleNamespace(pts=x) for x in pts)
            with self.assertRaises(ValueError):choose_frame(container,stream,1.)


if __name__=='__main__':unittest.main()

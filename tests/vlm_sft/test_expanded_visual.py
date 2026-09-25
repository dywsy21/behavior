from contextlib import ExitStack, nullcontext
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/vlm_sft'))
import prepare_expanded_visual as prepare


class ExpandedVisualTests(unittest.TestCase):
    def episodes(self):
        return [{'episode_index':task*200+i,'task_index':task,'task_instance_id':i+1,'length':2000}
                for task in range(5) for i in range(200)]

    def test_instance_hash_split_is_order_independent_and_protects_old_groups(self):
        counts=json.loads(prepare.COUNTS.read_text());episodes=self.episodes()
        rows,identity=prepare.grouped_selection(episodes,counts)
        random.Random(6).shuffle(episodes)
        self.assertEqual((rows,identity),prepare.grouped_selection(episodes,counts))
        self.assertEqual(len(rows),480)
        blocked={tuple(x) for x in identity['protected_groups']};old={tuple(x) for x in identity['historical_train_groups']}
        self.assertFalse({(r['task'],r['instance']) for r in rows}&blocked)
        self.assertFalse({(r['task'],r['instance']) for r in rows if r['split']!='train'}&old)
        for task in range(5):
            self.assertEqual({s:sum(r['task']==task and r['split']==s for r in rows)
                              for s in ('train','validation','test')},{'train':80,'validation':8,'test':8})
            self.assertTrue({(task,i) for i in range(191,201)}<=blocked)
        self.assertTrue({(0,53),(1,185),(3,220)}<=blocked)
        with self.assertRaises(ValueError):prepare.grouped_selection(episodes[:-1],counts)

    def test_temporal_samples_exclude_quarantine_and_preserve_one_second_spacing(self):
        source={'episode':12,'frames':2000,'split':'train'}
        bad=[{'episode_index':12,'frame_start':300,'frame_end':600}]
        frames=prepare.sample_frames(source,bad)
        self.assertLessEqual(len(frames),32);self.assertEqual(frames[0],0);self.assertEqual(frames[-1],1999)
        self.assertTrue(all(b-a>=30 for a,b in zip(frames,frames[1:])))
        self.assertTrue(all(not 300<=f<=600 for f in frames))
        self.assertLessEqual(len(prepare.sample_frames({**source,'split':'test'},bad)),16)
        with self.assertRaises(ValueError):prepare.sample_frames(source,[{'episode_index':12,'frame_start':0,'frame_end':1999}])
        short=prepare.sample_frames({**source,'frames':150},[])
        self.assertLess(len(short),32);self.assertTrue(all(b-a>=30 for a,b in zip(short,short[1:])))

    def fixture(self,root):
        data=root/'data';data.mkdir();meta={'episode_index':0}
        for camera in prepare.CAMERAS.values():
            stem='videos/observation.rgb.'+camera
            meta.update({stem+'/chunk_index':0,stem+'/file_index':0,stem+'/from_timestamp':0.})
            video=data/stem/'chunk-000/file-000.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'original-video')
        sources=[{'split':'train','task':0,'instance':12,'episode':0,'frames':2000,'selected_frames':[0,999,1999]}]
        stream=SimpleNamespace(average_rate=Fraction(30),time_base=Fraction(1,30),codec_context=SimpleNamespace(thread_count=0))
        container=SimpleNamespace(streams=SimpleNamespace(video=[stream]))
        def opening(path):
            container.path=path;return nullcontext(container)
        def frame(container,stream,timestamp):
            size=720 if 'zed_link' in container.path else 480
            im=Image.new('RGB',(size,size),(int(timestamp)%255,55,77))
            return SimpleNamespace(pts=round(timestamp*30),to_image=lambda:im),timestamp
        return data,sources,{0:meta},SimpleNamespace(open=opening),frame

    def context(self,stack,root):
        data,sources,metadata,av,frame=self.fixture(root)
        stack.enter_context(patch.object(prepare,'ROOT',data))
        stack.enter_context(patch.object(prepare,'source_identity',return_value=(sources,metadata,{'exact':'identity'})))
        stack.enter_context(patch.dict(sys.modules,{'av':av}))
        stack.enter_context(patch.object(prepare,'choose_frame',side_effect=frame))
        stack.enter_context(patch.object(prepare.os,'sched_getaffinity',return_value=set(range(80))))
        stack.enter_context(patch.object(prepare.os,'sched_setaffinity'))
        stack.enter_context(patch.object(prepare.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
        stack.enter_context(patch.object(prepare.subprocess,'check_output',side_effect=lambda cmd,**kw:'' if 'status' in cmd else 'commit'))
        return sources

    def test_real_prepare_and_decode_validation_never_release_labels(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);self.context(stack,root);out=root/'out'
            result=prepare.prepare(out);self.assertEqual(result['image_count'],9)
            self.assertFalse(result['training_eligible']);self.assertEqual(result['new_model_calls'],0)
            self.assertEqual(result,prepare.validate_output(out))
            self.assertEqual(result['unique_exact_pixel_images'],6)
            with self.assertRaises(FileExistsError):prepare.prepare(out)
            seal=out/'complete.json';seal.unlink()
            with self.assertRaises(ValueError):prepare.validate_output(out)

    def test_failure_preserves_original_error_and_never_seals(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);self.context(stack,root);out=root/'out'
            stack.enter_context(patch.object(prepare,'choose_frame',side_effect=ValueError('VIDEO_ALIGNMENT_PRIMARY')))
            with self.assertRaisesRegex(ValueError,'VIDEO_ALIGNMENT_PRIMARY'):prepare.prepare(out)
            failure=json.loads((out/'failure.json').read_text())
            self.assertIn('VIDEO_ALIGNMENT_PRIMARY',failure['error']);self.assertFalse(failure['training_eligible'])
            self.assertFalse((out/'complete.json').exists())
            with self.assertRaises(ValueError):prepare.validate_output(out)

    def test_failure_receipt_waits_for_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);sources=self.context(stack,root);out=root/'out'
            stopped=threading.Event();second_entered=threading.Event()
            original_event=threading.Event;created=[]
            def event():
                created.append(1)
                return stopped if len(created)==1 else original_event()
            _,metadata,identity=prepare.source_identity.return_value
            sources[0]['selected_frames']=[0]
            sources.append({**sources[0],'episode':1,'selected_frames':[30]})
            metadata[1]=deepcopy(metadata[0])
            def decode(container,stream,timestamp):
                if timestamp==0:
                    self.assertTrue(second_entered.wait(2));raise ValueError('PRIMARY')
                second_entered.set();self.assertTrue(stopped.wait(2))
                return SimpleNamespace(pts=30,to_image=lambda:Image.new('RGB',(720,720))),1.
            stack.enter_context(patch.object(prepare.threading,'Event',side_effect=event))
            stack.enter_context(patch.object(prepare,'choose_frame',side_effect=decode))
            with self.assertRaisesRegex(ValueError,'PRIMARY'):prepare.prepare(out)
            files=list((out/'images').iterdir());failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(len(files),1);self.assertEqual(failure['image_count'],len(files))
            self.assertEqual(failure['image_bytes'],sum(p.stat().st_size for p in files))

    def test_failure_receipt_error_cannot_replace_primary(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);self.context(stack,root);out=root/'out';original=prepare.atomic_json
            def write(path,value):
                if path.name=='failure.json':raise OSError('SECONDARY')
                return original(path,value)
            stack.enter_context(patch.object(prepare,'atomic_json',side_effect=write))
            stack.enter_context(patch.object(prepare,'choose_frame',side_effect=ValueError('PRIMARY')))
            with self.assertRaisesRegex(ValueError,'PRIMARY'):prepare.prepare(out)
            self.assertFalse((out/'complete.json').exists())

    def test_package_rejects_sidecars_directories_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);self.context(stack,root);out=root/'out';prepare.prepare(out)
            sidecar=out/'teacher_labels.jsonl';sidecar.write_text('{"action":"forbidden"}')
            with self.assertRaises(ValueError):prepare.validate_output(out)
            sidecar.unlink();images=out/'images';moved=root/'elsewhere';images.rename(moved);images.symlink_to(moved)
            with self.assertRaises(ValueError):prepare.validate_output(out)
            images.unlink();moved.rename(images)
            extra=images/'nested';extra.mkdir()
            with self.assertRaises(ValueError):prepare.validate_output(out)
            extra.rmdir();manifest=out/'manifest.json';linked=root/'manifest.json';manifest.rename(linked);manifest.symlink_to(linked)
            with self.assertRaises(ValueError):prepare.validate_output(out)

    def test_sealed_final_timing_must_be_complete_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);self.context(stack,root);out=root/'out';prepare.prepare(out)
            manifest=out/'manifest.json';original=json.loads(manifest.read_text())
            for fields in ({'extraction_seconds':'bogus'},{'validation_seconds':-1},{'validation_seconds':float('nan')},
                           {'extraction_seconds':0.,'validation_seconds':0.},None):
                value=deepcopy(original)
                if fields is None:
                    del value['extraction_seconds'];del value['validation_seconds']
                else:value.update(fields)
                manifest.write_text(json.dumps(value))
                (out/'complete.json').write_text(json.dumps({'status':'RAW_COMPLETE_LABELS_PENDING',
                    'manifest_sha256':prepare.sha(manifest),'training_eligible':False}))
                with self.assertRaises(ValueError):prepare.validate_output(out)

    def test_changed_split_pixels_duplicate_hash_source_and_unreviewed_release_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);self.context(stack,root);out=root/'out';prepare.prepare(out)
            ledger=out/'images.jsonl';original=ledger.read_bytes();manifest=out/'manifest.json'
            result=json.loads(manifest.read_text());plan=out/'source_plan.json';initial_plan=plan.read_bytes()
            def reset_seal():
                (out/'complete.json').write_text(json.dumps({'status':'RAW_COMPLETE_LABELS_PENDING',
                    'manifest_sha256':prepare.sha(manifest),'training_eligible':False}))
            for kind in ('split','pixels','dhash','timestamp','label','teacher_label','action','success','pts','time_base'):
                rows=[json.loads(x) for x in original.splitlines()]
                if kind=='split':rows[0]['split']='test'
                elif kind=='label':rows[0]['training_eligible']=True
                elif kind in ('teacher_label','action','success'):rows[0][kind]='unauthorized'
                else:
                    key={'pixels':'raw_pixels_sha256','dhash':'difference_hash_256','timestamp':'requested_timestamp_s',
                         'pts':'source_frame_pts','time_base':'source_time_base'}[kind]
                    rows[0]['image_receipts']['head'][key]=999 if kind in ('timestamp','pts') else '123/1' if kind=='time_base' else 'c'*64
                ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows));changed={**result,'images_manifest_sha256':prepare.sha(ledger)}
                manifest.write_text(json.dumps(changed));reset_seal()
                with self.subTest(kind=kind),self.assertRaises(ValueError):prepare.validate_output(out)
            ledger.write_bytes(original);manifest.write_text(json.dumps(result));reset_seal()
            for kind in ('split','labels_created','input_modalities','teacher'):
                value=json.loads(initial_plan)
                if kind=='split':value['sources'][0]['split']='test'
                elif kind=='labels_created':value[kind]=True
                elif kind=='input_modalities':value[kind].append('privileged_state')
                else:value[kind]='forbidden'
                plan.write_text(json.dumps(value));manifest.write_text(json.dumps({**result,'source_plan_sha256':prepare.sha(plan)}));reset_seal()
                with self.subTest(kind=kind),self.assertRaises(ValueError):prepare.validate_output(out)


if __name__=='__main__':unittest.main()

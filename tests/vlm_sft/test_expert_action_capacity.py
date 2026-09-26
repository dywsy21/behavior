import hashlib
import json
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/vlm_sft'))
import audit_expert_action_capacity as audit


class CapacityTests(unittest.TestCase):
    def test_main_only_seals_unchanged_inputs(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for mutate in (False, True):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp)
                repo=root/'repo';(repo/'scripts/vlm_sft').mkdir(parents=True)
                for name in ('audit_expert_action_capacity.py','common.py','prepare.py','prepare_full_annotation.py'):
                    (repo/'scripts/vlm_sft'/name).write_bytes(b'# frozen test source\n')
                raw=root/'raw';raw.mkdir()
                source=[{'task':t,'instance':i,'episode':t*4+i,'frames':20,'split':'train'}
                        for t in range(5) for i in range(4)]
                (raw/'manifest.json').write_text('{}')
                (raw/'source_plan.json').write_text(json.dumps({'sources':source}))
                meta=root/'meta.parquet'
                pq.write_table(pa.Table.from_pylist([{'episode_index':s['episode'],'task_index':s['task'],
                    'task_instance_id':s['instance'],'length':s['frames']} for s in source]),meta)
                release=root/'release';release.mkdir()
                quarantine=release/'quarantine_ranges.parquet'
                pq.write_table(pa.table({'episode_index':pa.array([],type=pa.int64()),
                    'frame_start':pa.array([],type=pa.int64()),'frame_end':pa.array([],type=pa.int64())}),quarantine)
                label_root=root/'labels';label_root.mkdir()
                labels=label_root/'labels.parquet';labels.write_bytes(b'opaque pinned labels used by mocked scanner')
                digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
                plan=repo/'plan.json';plan.write_text(json.dumps({'metadata_sha256':
                    {name:digest(raw/name) for name in ('manifest.json','source_plan.json')},'review_files':{}}))
                counts=repo/'counts.json';counts.write_text(json.dumps({'source_identity':{
                    'labels_sha256':digest(labels),'quarantine_sha256':digest(quarantine),
                    'episode_meta_sha256':{str(meta):digest(meta)}}}))
                parquet=root/'source.parquet';parquet.write_bytes(b'fixed read receipt')
                stat=parquet.stat();out=root/'audit'
                def scanner(s,*args):
                    if mutate and s['episode']==0:
                        (repo/'scripts/vlm_sft/common.py').write_bytes(b'# changed mid-run\n')
                    return {**s,'training_eligible':False,'counts':{},'source_parquet':str(parquet),
                            'source_bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns}
                for key,value in {'REPO':repo,'PLAN':plan,'COUNTS':counts,'RAW':raw,'ROOT':root/'data',
                                  'LABELS':labels,'RELEASE':release,'__file__':str(repo/'scripts/vlm_sft/audit_expert_action_capacity.py')}.items():
                    stack.enter_context(patch.object(audit,key,value))
                stack.enter_context(patch.object(audit,'scan_episode',side_effect=scanner))
                stack.enter_context(patch.object(audit.subprocess,'check_output',side_effect=['','test_commit']))
                stack.enter_context(patch.object(sys,'argv',['audit','--output',str(out)]))
                stack.enter_context(redirect_stdout(StringIO()))
                if mutate:
                    with self.assertRaises(ValueError):audit.main()
                    self.assertFalse((out/'result.json').exists())
                    self.assertTrue((out/'failure.json').is_file())
                else:
                    audit.main()
                    result=json.loads((out/'result.json').read_text())
                    self.assertEqual(len(result['rows']),20)
                    self.assertFalse(result['training_eligible'])

    def test_verified_snapshot_and_midrun_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.json'
            content=b'{"original":true}'
            path.write_bytes(content)
            pins={}
            saved=audit.snapshot(path,pins,hashlib.sha256(content).hexdigest())
            path.write_bytes(b'{"replaced":true}')
            self.assertEqual(audit.strict_json(saved),{'original':True})
            with self.assertRaises(ValueError):
                audit.verify_completion(pins,[],time.monotonic(),[])

    def test_readonly_source_alias_is_content_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'actual';path.write_bytes(b'original')
            alias=Path(tmp)/'alias';alias.symlink_to(path)
            pins={};content=audit.snapshot(alias,pins,hashlib.sha256(b'original').hexdigest())
            self.assertEqual(content,b'original')
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):audit.snapshot(alias,{},pins[str(alias)])

    def test_metadata_duplicates_and_identity_mismatch_rejected(self):
        source={'task':2,'instance':3,'episode':4,'frames':20}
        row={'task_index':2,'task_instance_id':3,'episode_index':4,'length':20}
        self.assertEqual(audit.index_metadata([row],[source]),{4:row})
        with self.assertRaises(ValueError):audit.index_metadata([row,row],[source])
        for key in ('task_index','task_instance_id','episode_index','length'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit.index_metadata([{**row,key:99}],[source])

    def test_final_budget_rejects_overrun(self):
        with patch.object(audit.time,'monotonic',return_value=901):
            with self.assertRaises(TimeoutError):audit.verify_completion({},[],0,[])

    def test_duplicate_or_wrong_result_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source';path.write_bytes(b'fixed');stat=path.stat()
            selected=[{'task':t,'instance':i,'episode':t*4+i,'frames':20} for t in range(5) for i in range(4)]
            rows=[{**s,'training_eligible':False,'source_parquet':str(path),'source_bytes':stat.st_size,
                   'source_mtime_ns':stat.st_mtime_ns} for s in selected]
            audit.verify_completion({},rows,time.monotonic(),selected)
            for changed in ([rows[0]]*20,[{**r,'instance':99} for r in rows],
                            [{**r,'training_eligible':True} for r in rows]):
                with self.assertRaises(ValueError):audit.verify_completion({},changed,time.monotonic(),selected)

    def test_selection_train_only_protected_and_deterministic(self):
        plan={'sources':[{'task':t,'instance':i,'split':'train' if i<8 else 'test'} for t in range(5) for i in range(10)]}
        review=[{'protect_groups_in_future_student_releases':[[t,0] for t in range(5)]}]
        first=audit.selected_sources(plan,review)
        self.assertEqual(len(first),20)
        self.assertTrue(all(r['split']=='train' and r['instance']!=0 for r in first))
        plan['sources'].reverse()
        self.assertEqual(first,audit.selected_sources(plan,review))

    def label(self,branch='low'):
        return {'frame_index':0,'memlite_branch':branch,'source_kind':'original_demo',
                'low_action_supervision_mask':True,'action_horizon_end':32,'segment_end':32,
                'active_skills_semantic_json':'[{"verb":"GRASP","target":"radio","arm":"right"}]'}

    def test_low_branch_never_silently_overwritten_by_high(self):
        frames,_=audit.released_windows([self.label(),self.label('high')],18,[])
        self.assertEqual(frames,[0])
        with self.assertRaises(ValueError):audit.released_windows([self.label(),self.label()],18,[])

    def test_quarantine_and_segment_boundary_excluded(self):
        row=self.label()
        self.assertEqual(audit.released_windows([row],18,[(16,17)])[0],[])
        row['segment_end']=16
        self.assertEqual(audit.released_windows([row],18,[])[0],[])

    def test_endpoint_scale_check_does_not_certify_base_or_gripper(self):
        self.assertIsNone(audit.endpoint_compatible('BASE_FORWARD',{}))
        self.assertIsNone(audit.endpoint_compatible('RIGHT_CLOSE',{}))
        evidence={'delta_eef_base_m':{'right':[.01,0,0]}}
        self.assertTrue(audit.endpoint_compatible('RIGHT_FORWARD',evidence))
        evidence['delta_eef_base_m']['right']=[.03,0,0]
        self.assertFalse(audit.endpoint_compatible('RIGHT_FORWARD',evidence))


if __name__=='__main__':unittest.main()

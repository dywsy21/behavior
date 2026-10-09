"""Synthetic structural tests only; real media is separately read on A800."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from recovery_corpus import canonical,digest,file_sha
spec=importlib.util.spec_from_file_location('reviewed_union_tool',BASE/'tools/merge_reviewed_recovery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class UnionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.protected=self.root/'protected.json'
        self.protected.write_text(json.dumps({'groups':['t:99']}))

    def unit(self,index,split='train'):
        root=self.root/f'source{index}';corpus=root/'corpus'
        for folder in ('raw','audit','history'): (corpus/folder).mkdir(parents=True)
        def write(path,value): path.write_text(json.dumps(value)+'\n')
        archive=corpus/'raw/clip.zip';archive.write_bytes(b'structural-fixture-not-real-zip')
        ep=dict(run='run',episode_id=f'ep{index}',task='t',instance_id=index,split=split)
        inv=[dict(path='clip.zip',sha256=file_sha(archive),episode=ep)]
        row=dict(sample_id=f's{index}',source_episode=['run',f'ep{index}'],source_group=f't:{index}',
            task='t',instance_id=index,split=split,control_step=16,label_status='candidate_pending_semantic_review',
            actor_input=dict(issued_skills_semantic_json='[{"verb":"GRASP"}]',parent_goal='g',rgb=dict(archive='clip.zip')),
            label_audit=dict(full_executed_32_step_target_available=False,outcome=dict(value='FAILED')))
        write(corpus/'audit/inventory.json',inv);write(corpus/'audit/anchors.jsonl',row)
        write(corpus/'history/contexts.jsonl',dict(sample_id=f's{index}',source_group=f't:{index}'))
        write(corpus/'history/receipt.json',dict(inventory_sha256=digest(inv),
            anchors_sha256=file_sha(corpus/'audit/anchors.jsonl'),contexts_sha256=file_sha(corpus/'history/contexts.jsonl')))
        write(corpus/'audit/summary.json',dict(source_manifest_supplied=True,protected_groups_supplied=True,
            bindings_sha256='b'*64,source_manifest_sha256='a'*64,conflicting_evidence=0,quarantined=0,
            inventory_sha256=digest(inv),protected_groups_sha256=file_sha(self.protected)))
        media=root/'review.json';write(media,dict(synthetic=True))
        approval=dict(sample_id=f's{index}',pool='outcome',reviewer='SYNTHETIC TEST ONLY',event_id=f'e{index}',
            reviewed_start=0,reviewed_end=32,source_group=f't:{index}',
            evidence=[dict(path='review.json',sha256=file_sha(media),kind=k) for k in ('original_media_review','physical_semantic_review')],
            label=dict(value='FAILED',member_index=0,available_control_step=16,evidence_end_control_step=16))
        app=root/'approvals.json';write(app,dict(schema='recovery_sample_approvals_v1',inventory_sha256=digest(inv),
            anchors_sha256=file_sha(corpus/'audit/anchors.jsonl'),approvals=[approval]))
        return dict(corpus=str(corpus),evidence_root=str(root),approvals=[str(app)])

    def build(self,units):
        with patch.object(module,'CandidateArchiveReader') as reader:
            result=module.build_union(dict(evidence_root=str(self.root),protected_groups=str(self.protected),units=units),self.root/'union')
            self.assertEqual(reader.return_value.observation.call_count,result['observations_read'])
            self.assertEqual(reader.return_value.observation_and_actions.call_count,0)
        return result

    def test_exact_approval_relocation_does_not_invent_gate(self):
        result=self.build([self.unit(1),self.unit(2,'dev')])
        self.assertEqual(result['approved_rows'],dict(outcome=2,planner=0,action=0))
        self.assertFalse(result['pools']['outcome']['training_ready'])
        rows=[json.loads(s) for s in (self.root/'union/admission/outcome.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['candidate']['actor_input']['rgb']['archive'],'unit_00/clip.zip')
        self.assertEqual(rows[0]['approval']['evidence'][0]['path'],'source1/review.json')
        self.assertEqual(file_sha(self.root/'union/raw/unit_00/clip.zip'),file_sha(self.root/'source1/corpus/raw/clip.zip'))
        self.assertEqual(result['new_semantic_approvals'],0)

    def test_duplicate_snapshot_episode_rejected(self):
        unit=self.unit(1)
        with self.assertRaisesRegex(ValueError,'Duplicate episode'):self.build([unit,deepcopy(unit)])

    def test_tampered_review_and_changed_history_rejected(self):
        unit=self.unit(1);path=self.root/'source1/review.json';original=path.read_bytes()
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.build([unit])
        path.write_bytes(original)
        (self.root/'source1/corpus/history/contexts.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'causal history'):self.build([unit])

    def test_duplicate_objective_and_protected_group_rejected(self):
        unit=self.unit(1);unit['approvals']*=2
        with self.assertRaisesRegex(ValueError,'Duplicate objective'):self.build([unit])
        protected=self.unit(99)
        with self.assertRaisesRegex(ValueError,'held-out'):self.build([protected])


if __name__=='__main__':unittest.main()

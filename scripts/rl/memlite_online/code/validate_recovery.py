"""CPU-only structural/clock/content verification; NOT semantic human QA."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image


def validate_archive(path):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist())>512*2**20:
            raise ValueError('Candidate archive unexpectedly large')
        if len(archive.namelist())!=len(set(archive.namelist())):
            raise ValueError('Duplicate archive members')
        m=json.loads(archive.read('manifest.json'));raw=archive.read('transitions.jsonl')
        assert m['schema']=='recovery_candidate_v1' and m['label_kind']=='on_policy_candidate'
        assert m['bc_eligible'] is False and m['human_review']=='pending' and m['synthetic_expert'] is False
        assert m['episode']['split']=='train'
        assert m['transitions_sha256']==hashlib.sha256(raw).hexdigest()
        rows=[json.loads(line) for line in raw.splitlines()]
        assert rows and len(rows)<=1024
        assert [r['control_step'] for r in rows]==list(range(m['start_control_step'],m['end_control_step']))
        assert m['events']
        images=0
        for step,anchor in m['rgb_anchors'].items():
            assert set(anchor['sha256'])=={'head_rgb','left_wrist_rgb','right_wrist_rgb'}
            for camera,expected in anchor['sha256'].items():
                raw_image=archive.read(f'rgb/{int(step):08d}/{camera}.jpg')
                assert hashlib.sha256(raw_image).hexdigest()==expected
                with Image.open(io.BytesIO(raw_image)) as image:
                    image.load();shape=anchor['dimensions'][camera]
                    assert image.mode=='RGB' and shape==[3,image.height,image.width]
                images+=1
        for index,r in enumerate(rows):
            assert r['simulator_apply_ack'] is True
            for key,width in [('action_executed_raw23',23),('proprio_before',61),('proprio_after',61)]:
                value=np.asarray(r[key]);assert value.shape==(width,) and np.isfinite(value).all()
            assert 0<=r['control_step']-r['rgb_anchor_control_step']<16
            assert str(r['rgb_anchor_control_step']) in m['rgb_anchors']
            assert r['chunk_start_control_step']==r['rgb_anchor_control_step']
            assert r['experience_id']>=0 and r['policy_update']>=0
            assert r['context']['context_id']
            assert json.loads(r['context']['active_skills_semantic_json'])
            if index:
                assert rows[index-1]['proprio_after']==r['proprio_before']
                assert not (rows[index-1]['terminated'] or rows[index-1]['truncated'])
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    controls=len(rows),images=images,episode=m['episode'],events=m['events'],
                    structural_validation='passed',human_review='pending',bc_eligible=False)

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
        offline_teacher=m['schema']=='recovery_teacher_candidate_v2'
        if offline_teacher:
            assert m['label_kind']=='offline_local_teacher_candidate'
            assert m['episode']['actor_model_used'] is False
        else:
            assert m['schema']=='recovery_candidate_v1' and m['label_kind']=='on_policy_candidate'
            assert m['synthetic_expert'] is False
        assert m['bc_eligible'] is False and m['human_review']=='pending'
        assert m['episode']['split']=='train'
        assert m['transitions_sha256']==hashlib.sha256(raw).hexdigest()
        rows=[json.loads(line) for line in raw.splitlines()]
        assert rows and (offline_teacher or len(rows)<=1024)
        if offline_teacher:
            from recovery_teacher_corpus import validate_branch
            validate_branch(rows,m['branch_evidence'],m['plans'])
            if 'terminal_observation' in m:
                from recovery_local_teacher import validate_terminal_observation
                from recovery_terminal_corpus import normalized_terminal
                original_raw=archive.read('terminal-observation.original.json')
                assert hashlib.sha256(original_raw).hexdigest()==m['branch_evidence']['terminal_observation_sha256']
                original=json.loads(original_raw);terminal=m['terminal_observation']
                binding=rows[-1]['physical_audit']['entity_bindings']
                assert terminal==normalized_terminal(original,binding)
                validate_terminal_observation(terminal,rows,m['branch_evidence']['arm'])
                assert terminal['context']==m['plans'][-1]
                assert terminal['last_applied_control_step']==rows[-1]['control_step']
                assert terminal['control_step']==m['end_control_step']
                assert m['rgb_anchors'][str(terminal['control_step'])]['observation_only'] is True
                assert m['rgb_anchors'][str(terminal['control_step'])]['sha256']=={
                    k:v['sha256'] for k,v in terminal['images'].items()}
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

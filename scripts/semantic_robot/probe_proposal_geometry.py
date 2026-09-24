"""H60: local CPU-only inspection of all H58 boxes; never a policy or new inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
import probe_finite_localization as source
from semantic_robot.v2.proposal_geometry import diagnose_box

INPUTS=REPO/'artifacts/agentic-vlm-goal-20260918/h60_public_inputs_v2'
DETECTIONS=REPO/'artifacts/agentic-vlm-goal-20260918/h58_text_detector_bundle_v1/result.json'
OUTPUT=REPO/'artifacts/agentic-vlm-goal-20260918/h60_proposal_geometry_v1'
INPUT_SPEC=REPO/'configs/semantic_robot/h51_native_grounding_probe.json'
INPUT_SHA='76b0e7e366ff921c1dbf30b57d280833e3476a4c7698d5979684b45089347f61'
DETECTION_SHA='7aba22853b9458f06205abcdc3543ef166378a4dc7648bfc590ef0776854668c'


def checked_inputs():
    for path,digest in ((INPUT_SPEC,INPUT_SHA),(DETECTIONS,DETECTION_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Exact frozen H51 inputs and H58 outputs required')
    spec=json.loads(INPUT_SPEC.read_text());source.validate_spec(spec)
    result=json.loads(DETECTIONS.read_text())
    expected={case['id']+'__'+view for case in spec['cases'] for view in ('head','left_wrist','right_wrist')}
    if (result['status']!='completed' or result['cuda_initialized'] is not False or
            len(result['queries'])!=12 or {q['id'] for q in result['queries']}!=expected or
            sum(len(q['detections']) for q in result['queries'])!=10):
        raise ValueError('All 12 queries and 10 original predictions required')
    prepared=[]
    for case in spec['cases']:
        root=INPUTS/Path(case['source_run']).relative_to('/mnt/nvme_tmp')
        arguments,binding=source.load_case(case,root)
        rows=[q for q in result['queries'] if q['id'].startswith(case['id']+'__')]
        if len(rows)!=3 or any(q['frame_binding']!=binding for q in rows):
            raise ValueError('Detector result and public frame do not match')
        for row in rows:
            if hashlib.sha256(arguments['raw'][row['view']].tobytes()).hexdigest()!=row['raw_pixels_sha256']:
                raise ValueError('RAW pixel mismatch')
        prepared.append((arguments,rows))
    return prepared


def save(record):
    pending=OUTPUT/'result.pending'
    pending.write_text(json.dumps(record,indent=2,allow_nan=False))
    pending.replace(OUTPUT/'result.json')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only environment required')
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean fixed source required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    started=time.monotonic()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    OUTPUT.mkdir(exist_ok=False)
    record={'status':'running','source_commit':commit,'pid':os.getpid(),'queries':[],
        'input_spec_sha256':INPUT_SHA,'detector_result_sha256':DETECTION_SHA,
        'model_calls':0,'simulator_resets':0,'training_steps':0,'success_rate':None,'not_success_rate':True}
    save(record)
    try:
        for arguments,rows in checked_inputs():
            arguments={k:v for k,v in arguments.items() if k!='views'}
            for row in rows:
                if time.monotonic()-started>=60:raise TimeoutError('60s geometry diagnostic budget')
                results=[diagnose_box(**arguments,view=row['view'],box=d['box_xyxy_px']) for d in row['detections']]
                record['queries'].append({'id':row['id'],'view':row['view'],'detections':row['detections'],'geometry':results})
                save(record)
        if len(record['queries'])!=12 or any(k in sys.modules for k in ('torch','isaacsim','omnigibson')):
            raise ValueError('CPU-only complete diagnostic required')
        record.update(status='completed',seconds=time.monotonic()-started)
        save(record)
    except BaseException as error:
        record.update(status='failed',error=repr(error),seconds=time.monotonic()-started);save(record)
        raise
    print(json.dumps({'status':record['status'],'seconds':record['seconds'],'queries':len(record['queries'])}))


if __name__=='__main__':main()

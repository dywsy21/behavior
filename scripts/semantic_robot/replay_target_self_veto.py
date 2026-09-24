"""H62: frozen model-answer / geometry replay, not model or task evaluation."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import probe_proposal_geometry as source
from semantic_robot.v2.finite_localization import SelectionReply, bind_frame
from semantic_robot.v2.grounding import GroundedEvidence, localize_target, unproject
from semantic_robot.v2.native_grounding import locate_native
from semantic_robot.v2.proposal_geometry import box_samples

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'artifacts/agentic-vlm-goal-20260918'
NATIVE=ROOT/'h51_shared_bundle_v1/h51_native_protocol_v1/result.json'
GEOMETRY=ROOT/'h60_proposal_geometry_v1/result.json'
NATIVE_SHA='3769f632eb2593237e8403d6e7d3b8c3d5ac75cdc5e87a5220a8414ff54723f6'
GEOMETRY_SHA='eb6b85bab705eb12a60a4aee5680fe034ef3f76a8d56877ac6105758a1e08d20'
OUTPUT=ROOT/'h62_target_self_replay_v1'


def frozen_json(path,digest):
    if path.is_symlink():raise ValueError('Frozen regular input required')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Frozen result SHA mismatch')
    return json.loads(data)


def replay_native(arguments,old,deadline):
    previous=old['native_point']['receipt']
    if bind_frame(**arguments)!=old['binding'] or previous['binding']!=old['binding']:
        raise ValueError('Original model answer requires its exact public frame')
    calls=[]
    def choose(request):
        if request.sha256!=previous['request_sha256'] or request.receipt()!=previous['request']:
            raise ValueError('Cached answer request mismatch')
        calls.append(request.sha256)
        return SelectionReply(request.sha256,previous['raw_text'])
    result=locate_native(**arguments,view=previous['view'],mode='point',choose=choose,deadline=deadline)
    if len(calls)!=1:raise ValueError('Exactly one cached answer required')
    return {'request_sha256':calls[0],'raw_text':previous['raw_text'],
        'historical_model_answer_count':1,'actual_model_calls':0,
        'previous_evidence':old['native_point']['evidence'],'new_evidence':asdict(result.point_evidence()),
        'previous_geometry':previous.get('geometry'),'new_geometry':result.receipt.get('geometry'),
        'reason':result.receipt['reason'],'semantic_correctness_not_established':True}


def replay_pixels(arguments,query,diagnostic,deadline):
    if (query['id']!=diagnostic['id'] or query['view']!=diagnostic['view'] or
            query['detections']!=diagnostic['detections'] or
            len(query['detections'])!=len(diagnostic['geometry'])):
        raise ValueError('All original detector boxes and diagnostic rows required')
    view=query['view'];model=arguments['model'];state=arguments['state']
    camera=model.spec['metadata']['cameras'][view]
    width,height=camera['width'],camera['height'];pixels=[]
    for detection,box in zip(query['detections'],diagnostic['geometry']):
        expected=box_samples(detection['box_xyxy_px'],width,height).tolist()
        if (box['frame_binding']!=query['frame_binding'] or box['view']!=view or
                box['box_xyxy_px']!=detection['box_xyxy_px'] or
                [row['pixel_xy'] for row in box['samples']]!=expected):
            raise ValueError('Original complete H60 pixel grid and frame binding required')
        for sample in box['samples']:
            if time.monotonic()>=deadline:raise TimeoutError('H62 60s wall deadline')
            x,y=sample['pixel_xy'];known_self=sample['on_chassis_surface']
            if type(known_self) is not bool:raise ValueError('H60 known self membership required')
            actual=unproject([[x,y]],[arguments['depths'][view][y,x]],camera['K'],
                             model.forward(state.q,'camera_'+view))[0]
            if not np.allclose(actual,sample['point_base_m'],rtol=0,atol=1e-10):
                raise ValueError('Sample no longer matches original sensor point')
            evidence=GroundedEvidence(visible=True,view=view,target_uv=(x/(width-1),y/(height-1)),
                enclosed=None,co_moving=None,supported=None,effect=None,hazard='none',
                note='Mechanical H60 geometry sample, not a model target prediction.')
            current=localize_target(evidence,arguments['depths'],model,state.q)
            pixels.append({'pixel_xy':[x,y],'original_chassis_sample':known_self,
                'current_valid':current['valid'],'current_reason':current['reason'],
                'current_point_base_m':current.get('point_base_m')})
    return {'id':query['id'],'view':view,'frame_binding':query['frame_binding'],'samples':pixels,
        'total':len(pixels),'original_self':sum(p['original_chassis_sample'] for p in pixels),
        'valid':sum(p['current_valid'] for p in pixels),
        'self_still_valid':sum(p['current_valid'] and p['original_chassis_sample'] for p in pixels),
        'reason_counts':dict(Counter(p['current_reason'] for p in pixels)),
        'samples_are_not_model_predictions_or_positive_semantic_labels':True}


def save(record):
    pending=OUTPUT/'result.pending'
    pending.write_text(json.dumps(record,indent=2,allow_nan=False));pending.replace(OUTPUT/'result.json')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only environment required')
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean fixed source required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    started=time.monotonic();deadline=started+60;OUTPUT.mkdir(exist_ok=False)
    record={'status':'running','source_commit':commit,'pid':os.getpid(),'cases':[],
        'source_result_sha256':{'h51':NATIVE_SHA,'h60':GEOMETRY_SHA},
        'model_calls':0,'simulator_resets':0,'control_steps':0,'training_steps':0,
        'success_rate':None,'not_success_rate':True}
    save(record)
    try:
        native=frozen_json(NATIVE,NATIVE_SHA);geometry=frozen_json(GEOMETRY,GEOMETRY_SHA)
        prepared=source.checked_inputs()
        case_ids={rows[0]['id'].rsplit('__',1)[0] for _,rows in prepared}
        if (native['status']!='complete' or geometry['status']!='completed' or
                len(native['cases'])!=4 or {c['id'] for c in native['cases']}!=case_ids or
                len(geometry['queries'])!=12 or {q['id'] for q in geometry['queries']}!=
                {r['id'] for _,rows in prepared for r in rows}):
            raise ValueError('Exactly the original four model cases and 12 geometry rows required')
        for arguments,queries in prepared:
            arguments={k:v for k,v in arguments.items() if k!='views'}
            case_id=queries[0]['id'].rsplit('__',1)[0]
            old=next(case for case in native['cases'] if case['id']==case_id)
            row={'id':case_id,'native_point':replay_native(arguments,old,deadline),'queries':[]}
            record['cases'].append(row);save(record)
            for query in queries:
                previous=next(q for q in geometry['queries'] if q['id']==query['id'])
                current=replay_pixels(arguments,query,previous,deadline)
                row['queries'].append(current);save(record)
                if current['self_still_valid']:raise AssertionError('Known robot self point escaped veto')
        rows=[q for case in record['cases'] for q in case['queries']]
        if (len(rows)!=12 or sum(q['total'] for q in rows)!=1440 or sum(q['original_self'] for q in rows)!=771 or
                any(k in sys.modules for k in ('torch','isaacsim','omnigibson'))):
            raise ValueError('Full CPU-only original sample count required')
        record.update(status='completed',seconds=time.monotonic()-started);save(record)
    except BaseException as error:
        record.update(status='failed',error=repr(error),seconds=time.monotonic()-started);save(record);raise
    print(json.dumps({'status':record['status'],'seconds':record['seconds'],'cases':len(record['cases'])}))


if __name__=='__main__':main()

"""TRAIN-only fixed-input, fixed-noise serial/batch/reordered GPU comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import atomic_json
from batched_engine import BatchedSFT


def difference(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    d=a-b
    return dict(shape=list(a.shape),max_abs=float(np.abs(d).max()),rms=float(np.sqrt(np.mean(d*d))),
                reference_rms=float(np.sqrt(np.mean(a*a))),finite=bool(np.isfinite(d).all()))


def audit(engine, snapshot):
    if any(meta.get('split')!='train_smoke' for meta in snapshot['episode_metadata']):
        raise ValueError('Tuning/numerical QA must not use public evaluation')
    indices=snapshot['indices'];observations=snapshot['observations']
    def restore(mode):
        engine.mode=mode;engine.capture=False;engine.requests=snapshot['requests']
        engine.slots=deepcopy(snapshot['slots']);engine.task=snapshot['task']
        engine.episode_metadata=deepcopy(snapshot['episode_metadata'])
        torch.cuda.set_rng_state(snapshot['cuda_rng'])
    restore('serial');started=time.monotonic()
    serial,serial_contexts=engine.infer_native(deepcopy(observations),indices)
    serial_seconds=time.monotonic()-started
    serial_projections=[deepcopy(engine.slots[i]['projection']) for i in indices]
    restore('batch');started=time.monotonic()
    batched,batch_contexts=engine.infer_native(deepcopy(observations),indices)
    batch_seconds=time.monotonic()-started
    # Isolate low numerical error from any high-level discrete choice changes.
    for obs in observations:obs['task']=engine.task
    noise=[];raw_serial=[];normalized_serial=[]
    for obs,projection in zip(observations,serial_projections,strict=True):
        raw,normalized=engine.low_batch([obs],[projection],capture_noise=noise)
        raw_serial.append(raw);normalized_serial.append(normalized)
    fixed=torch.cat(noise,dim=0)
    raw_serial=np.concatenate(raw_serial);normalized_serial=np.concatenate(normalized_serial)
    raw_batch,normalized_batch=engine.low_batch(observations,serial_projections,fixed_noise=fixed)
    order=list(reversed(range(len(indices))));inverse=np.argsort(order)
    raw_permuted,normalized_permuted=engine.low_batch([observations[i] for i in order],
        [serial_projections[i] for i in order],fixed_noise=fixed[order])
    result=dict(request=snapshot['requests'],indices=indices,serial_seconds=serial_seconds,
        task=snapshot['task'],session_generation=snapshot.get('session_generation'),
        batch_seconds=batch_seconds,high_context_equal=[a==b for a,b in zip(serial_contexts,batch_contexts)],
        capture_replay=difference(snapshot['actions'],
            serial if snapshot.get('inference_mode')=='serial' else batched),
        serial_contexts=serial_contexts,batch_contexts=batch_contexts,
        full_actions=difference(serial,batched),fixed_context_raw23=difference(raw_serial,raw_batch),
        fixed_context_normalized27=difference(normalized_serial,normalized_batch),
        permuted_raw23=difference(raw_batch,raw_permuted[inverse]),
        permuted_normalized27=difference(normalized_batch,normalized_permuted[inverse]),
        raw_dimension_max_abs=np.max(np.abs(raw_serial-raw_batch),axis=(0,1)).tolist())
    # Keep the original gate unchanged while diagnosing any rejection. Real
    # controls and padding, as well as executed/future horizons, are distinct.
    delta=np.abs(normalized_serial-normalized_batch)
    valid=[i for i in range(27) if i not in (7,8,17,18)]
    location=np.unravel_index(np.argmax(delta),delta.shape)
    result['normalized_error_detail']=dict(
        argmax=[int(x) for x in location],
        serial_value=float(normalized_serial[location]),batch_value=float(normalized_batch[location]),
        per_dimension_max=np.max(delta,axis=(0,1)).tolist(),
        executed_real=difference(normalized_serial[:,:16,valid],normalized_batch[:,:16,valid]),
        future_real=difference(normalized_serial[:,16:,valid],normalized_batch[:,16:,valid]),
        padding=difference(normalized_serial[:,:,[7,8,17,18]],normalized_batch[:,:,[7,8,17,18]]),
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
    # Engineering gate, not a claim of policy/success-rate equivalence.
    result['engineering_gate_passed']=(all(result['high_context_equal']) and
        result['capture_replay']['max_abs']<=.005 and
        result['fixed_context_normalized27']['finite'] and
        result['fixed_context_normalized27']['max_abs']<=.05 and
        result['fixed_context_normalized27']['rms']<=.005 and
        result['permuted_normalized27']['max_abs']<=.005)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    engine=BatchedSFT(a.output/'no_checkpoints')
    results=[]
    for path in a.inputs:
        result=audit(engine,torch.load(path,map_location='cpu',weights_only=False))
        result['input']=str(path);results.append(result)
        atomic_json(a.output/'audit.json',dict(results=results,complete=False))
    receipt=engine.verify()
    atomic_json(a.output/'audit.json',dict(results=results,complete=True,verification=receipt,
        engineering_gate_passed=all(r['engineering_gate_passed'] for r in results)))
    print(json.dumps(dict(output=str(a.output),passed=all(r['engineering_gate_passed'] for r in results))))


if __name__=='__main__':main()

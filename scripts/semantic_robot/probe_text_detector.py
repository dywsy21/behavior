"""One CPU-only, frozen 12-RAW text detector check; no simulator or actor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
import probe_finite_localization as source
import probe_shared_vlm as shared

SPEC = REPO / 'configs/semantic_robot/h58_text_detector_cpu.json'
PYTHON = Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python')
OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h58_text_detector_cpu_v1')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_vlm_runtime_20260924/h58_cpu')
INPUT_SPEC = 'configs/semantic_robot/h51_native_grounding_probe.json'
INPUT_SHA = '76b0e7e366ff921c1dbf30b57d280833e3476a4c7698d5979684b45089347f61'
MANIFEST_SHA = '046922670d986c772d797b0276079ec530456db2afe73fe198aec78e9e9104f7'


def read_spec():
    spec = json.loads(SPEC.read_text())
    expected = dict(experiment='H58-text-detector-cpu-v1', input_spec=INPUT_SPEC,
        input_spec_sha256=INPUT_SHA, views=['head','left_wrist','right_wrist'],
        max_calls=12, max_seconds=600, cpu_cores=[68,69,70,71], threads=4,
        box_threshold=.4, text_threshold=.3, torch='2.7.1+cu128', transformers='4.57.1',
        device='cpu', dtype='float32', seed=17, not_success_rate=True,
        training_steps=0, simulator_resets=0,
        revision='a2bb814dd30d776dcf7e30523b00659f4f141c71',
        model='/mnt/nvme_tmp/robodojo_grounding_models_20260924/grounding-dino-tiny_a2bb814')
    if set(spec) != set(expected) | {'model_files'} or any(
            type(spec[k]) is not type(v) or spec[k] != v for k,v in expected.items()):
        raise ValueError('Exact registered CPU probe required')
    manifest = json.dumps(spec['model_files'], sort_keys=True, separators=(',', ':'))
    if hashlib.sha256(manifest.encode()).hexdigest() != MANIFEST_SHA:
        raise ValueError('Exact nine-file official model identity required')
    if shared.sha(REPO / INPUT_SPEC) != INPUT_SHA:
        raise ValueError('Frozen public input specification changed')
    return spec


def query_text(target):
    if not isinstance(target, str) or not 1 <= len(target.strip()) <= 300:
        raise ValueError('Public target description required')
    return target.strip().lower().rstrip('.').strip() + '.'


def prepare_inputs(spec):
    original = json.loads((REPO / INPUT_SPEC).read_text())
    source.validate_spec(original)
    queries = []
    for case in original['cases']:
        data, binding = source.load_case(case)
        for view in spec['views']:
            raw = data['raw'][view]
            if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
                raise ValueError('Original uint8 RGB required')
            queries.append({'id': case['id']+'__'+view, 'view': view,
                'text': query_text(data['goal'].target), 'image': Image.fromarray(raw),
                'frame_binding': binding, 'raw_pixels_sha256': hashlib.sha256(raw.tobytes()).hexdigest()})
    if len(queries) != spec['max_calls'] or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Exactly 12 unique frozen target/view queries required')
    return queries


def detections(result):
    """Keep all thresholded predictions, including false positives and outside boxes."""
    scores, boxes, labels = result['scores'], result['boxes'], result['text_labels']
    if hasattr(scores, 'detach'): scores = scores.detach().cpu().numpy()
    if hasattr(boxes, 'detach'): boxes = boxes.detach().cpu().numpy()
    scores, boxes = np.asarray(scores), np.asarray(boxes)
    if (scores.ndim != 1 or boxes.shape != (len(scores),4) or len(labels) != len(scores) or
            len(scores) > 900 or not np.isfinite(scores).all() or not np.isfinite(boxes).all() or
            np.any(scores < 0) or np.any(scores > 1) or
            np.any(boxes[:,2:] <= boxes[:,:2]) or any(type(label) is not str for label in labels)):
        raise ValueError('Malformed detector output; never repair or cherry-pick')
    return [{'box_xyxy_px':box.astype(float).tolist(), 'score':float(score), 'text_label':label}
            for box,score,label in zip(boxes,scores,labels)]


def identity():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError('Frozen existing interpreter required; no environment installation')
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean immutable source required')
    return subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()


def cpu_environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(REPO/'src'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        HF_HOME=str(RUNTIME/'hf'), TORCH_HOME=str(RUNTIME/'torch'), XDG_CACHE_HOME=str(RUNTIME/'cache'))
    return env


def write(name, value):
    pending = OUTPUT / (name+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False))
    pending.replace(OUTPUT/name)


def launch():
    spec, commit = read_spec(), identity()
    if OUTPUT.exists() or RUNTIME.exists():
        raise FileExistsError('Never resubmit an existing H58 run')
    shared.validate_model_files(Path(spec['model']), spec['model_files'])
    if shutil.disk_usage(OUTPUT.parent).free < 80*1024**3:
        raise ValueError('Preserve output disk reserve')
    prepare_inputs(spec)  # All public input hashes before reserving the one run.
    OUTPUT.mkdir(parents=True, exist_ok=False); RUNTIME.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(24)
    env = cpu_environment(); env['H58_LAUNCH_TOKEN'] = token
    command = [str(PYTHON),str(Path(__file__).resolve()),'--supervisor']
    receipt = {'source_commit':commit, 'spec_sha256':shared.sha(SPEC), 'command':command,
        'token_sha256':hashlib.sha256(token.encode()).hexdigest(),
        'utc':datetime.now(timezone.utc).isoformat(), 'status':'reserved',
        'model_calls':12, 'seconds':600, 'cleanup_seconds':15,
        'gpu_allocations':0, 'cpu_cores':spec['cpu_cores']}
    write('launch.json',receipt)
    try:
        with (OUTPUT/'supervisor.log').open('x') as log:
            child = subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException as error:
        receipt.update(status='launch_failed',error=repr(error))
        write('launch.json',receipt)
        raise
    receipt.update(status='supervisor_started',supervisor_pid=child.pid)
    write('launch.json',receipt)
    print(json.dumps(receipt),flush=True)


def claim_stage(mode, commit):
    if mode not in ('supervisor', 'worker'):
        raise ValueError('Known stage required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('CPU-only and offline worker required')
    # The parent writes its child's PID just after Popen. Wait only for that atomic receipt.
    deadline = time.monotonic() + 5
    while True:
        launch_record = json.loads((OUTPUT/'launch.json').read_text())
        if launch_record.get('status') != 'reserved' or time.monotonic() >= deadline:
            break
        time.sleep(.05)
    token = os.environ.get('H58_LAUNCH_TOKEN','')
    if (len(token) != 48 or hashlib.sha256(token.encode()).hexdigest() != launch_record['token_sha256'] or
            commit != launch_record['source_commit'] or shared.sha(SPEC) != launch_record['spec_sha256'] or
            launch_record.get('status') != 'supervisor_started'):
        raise ValueError('Source/spec/single-use reservation mismatch')
    expected = os.getpid() if mode == 'supervisor' else os.getppid()
    if launch_record.get('supervisor_pid') != expected:
        raise ValueError('Only the reserved supervisor and its own worker may run')
    if mode == 'worker':
        parent = json.loads((OUTPUT/'supervisor.claim.json').read_text())
        if parent.get('pid') != expected or parent.get('source_commit') != commit:
            raise ValueError('Live parent stage identity mismatch')
    with (OUTPUT/(mode+'.claim.json')).open('x') as f:
        json.dump({'pid':os.getpid(),'ppid':os.getppid(),'source_commit':commit},f)


def check_completed(record, child, commit):
    original = json.loads((REPO / INPUT_SPEC).read_text())
    expected = {c['id']+'__'+view for c in original['cases']
                for view in ('head','left_wrist','right_wrist')}
    queries = record.get('queries', [])
    if (child.returncode != 0 or record.get('status') != 'completed' or
            record.get('pid') != child.pid or record.get('source_commit') != commit or
            record.get('spec_sha256') != shared.sha(SPEC) or
            record.get('cuda_initialized') is not False or len(queries) != 12 or
            {q.get('id') for q in queries} != expected):
        raise ValueError('Worker exit alone is not a complete CPU probe')


def supervisor():
    spec, commit = read_spec(), identity()
    claim_stage('supervisor', commit)
    started = time.monotonic()
    record = {'status':'running','pid':os.getpid(),'source_commit':commit,
        'spec_sha256':shared.sha(SPEC),'wall_limit_seconds':600,'cleanup_limit_seconds':15,
        'gpu_allocations':0,'not_success_rate':True,'worker_pid':None}
    child = None
    write('supervisor.json',record)
    def interrupted(signum, frame):
        raise InterruptedError('H58 supervisor signal ' + str(signum))
    previous = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            with (OUTPUT/'worker.log').open('x') as log:
                child = subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--worker'],
                    cwd=REPO,env=cpu_environment(),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['worker_pid'] = child.pid
        finally:
            # Pending TERM is delivered only after this exact child is owned.
            signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        write('supervisor.json',record)
        child.wait(timeout=max(.001, spec['max_seconds']-(time.monotonic()-started)))
        result = json.loads((OUTPUT/'result.json').read_text())
        check_completed(result, child, commit)
        record.update(status='completed',model_calls=len(result['queries']))
    except subprocess.TimeoutExpired as error:
        record.update(status='timed_out',error=repr(error))
    except BaseException as error:
        record.update(status='failed',error=repr(error))
    finally:
        # Only this Popen child is signalled. No GPU process discovery or group-wide kill.
        # A second external stop must not interrupt the bounded reap or terminal write.
        for sig in previous: signal.signal(sig,signal.SIG_IGN)
        try:
            if child is not None and child.poll() is None:
                try:
                    child.terminate(); record['terminate_sent'] = True
                    child.wait(timeout=7.5)
                except subprocess.TimeoutExpired:
                    child.kill(); record['kill_sent'] = True
                    child.wait(timeout=7.5)
        except BaseException as error:
            record.update(status='cleanup_failed',cleanup_error=repr(error))
        finally:
            record.update(seconds=time.monotonic()-started,
                exit_code=child.returncode if child is not None else None,
                worker_reaped=child is None or child.returncode is not None,
                finished_utc=datetime.now(timezone.utc).isoformat())
            try:
                result_path = OUTPUT/'result.json'
                if result_path.is_file():
                    record['worker_result_sha256'] = shared.sha(result_path)
            except BaseException as error:
                record.update(status='failed',result_digest_error=repr(error))
            try:
                write('supervisor.json',record)
            finally:
                for sig,handler in previous.items(): signal.signal(sig,handler)
    return 0 if record['status'] == 'completed' else 1


def worker():
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    spec, commit = read_spec(), identity()
    claim_stage('worker', commit)
    started = time.monotonic()
    record = {'status':'running','pid':os.getpid(),'source_commit':commit,'spec_sha256':shared.sha(SPEC),
        'success_rate':None,'not_success_rate':True,'training_steps':0,'simulator_resets':0,
        'gpu_allocations':0,'queries':[],'revision':spec['revision']}
    write('result.json',record)
    try:
        cores = set(spec['cpu_cores'])
        if not cores <= os.sched_getaffinity(0): raise ValueError('CPU reservation unavailable')
        os.sched_setaffinity(0,cores); os.nice(10)
        shared.validate_model_files(Path(spec['model']),spec['model_files'])
        queries = prepare_inputs(spec)
        import torch
        import transformers
        from transformers import GroundingDinoConfig, GroundingDinoForObjectDetection, GroundingDinoProcessor
        if torch.__version__ != spec['torch'] or transformers.__version__ != spec['transformers']:
            raise ValueError('Frozen Torch/Transformers version mismatch')
        torch.set_num_threads(spec['threads']); torch.set_num_interop_threads(1); torch.manual_seed(spec['seed'])
        config = GroundingDinoConfig.from_pretrained(spec['model'],local_files_only=True)
        config.disable_custom_kernels = True
        processor = GroundingDinoProcessor.from_pretrained(spec['model'],local_files_only=True,use_fast=False)
        model = GroundingDinoForObjectDetection.from_pretrained(spec['model'],config=config,
            use_safetensors=True,local_files_only=True).eval()
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('All parameters must be FP32 CPU')
        record.update(load_seconds=time.monotonic()-started,processor=type(processor.image_processor).__name__,
            processor_config=processor.image_processor.to_dict(),disable_custom_kernels=model.config.disable_custom_kernels,
            parameters=sum(p.numel() for p in model.parameters()))
        write('result.json',record)
        for i,query in enumerate(queries):
            if time.monotonic()-started >= spec['max_seconds']: raise TimeoutError('CPU probe budget')
            directory = OUTPUT/query['id']; directory.mkdir()
            query['image'].save(directory/'RAW.png')
            call_started = time.monotonic()
            inputs = processor(images=query['image'],text=query['text'],return_tensors='pt')
            with torch.inference_mode():
                output = model(**inputs)
                post = processor.post_process_grounded_object_detection(output,inputs.input_ids,
                    threshold=spec['box_threshold'],text_threshold=spec['text_threshold'],
                    target_sizes=[query['image'].size[::-1]])
            if len(post) != 1: raise ValueError('Only batch one permitted')
            prediction = {k:v for k,v in query.items() if k != 'image'}
            prediction.update(index=i,seconds=time.monotonic()-call_started,
                image_size=list(query['image'].size),input_shape=list(inputs['pixel_values'].shape),
                detections=detections(post[0]),box_units='original_RAW_pixels_xyxy',
                box_threshold=spec['box_threshold'],text_threshold=spec['text_threshold'],
                not_contact_or_completion=True)
            record['queries'].append(prediction)
            write('result.json',record)
            del output,post,inputs
        if len(record['queries']) != 12 or torch.cuda.is_initialized():
            raise ValueError('CPU-only 12-query contract violated')
        record.update(status='completed',seconds=time.monotonic()-started,cuda_initialized=False)
        write('result.json',record)
    except BaseException as error:
        record.update(status='failed',error=repr(error),seconds=time.monotonic()-started)
        write('result.json',record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','launch','supervisor','worker'): mode.add_argument('--'+name,action='store_true')
    args = parser.parse_args()
    if args.prepare:
        spec = read_spec()
        print(json.dumps({'queries':[{k:v for k,v in q.items() if k != 'image'} for q in prepare_inputs(spec)],
                          'model_calls':0,'simulator_resets':0},indent=2))
    elif args.launch: launch()
    elif args.supervisor: sys.exit(supervisor())
    else: worker()

"""Admit only identified live peer-simulator auxiliary CUDA contexts.

OmniGibson's render launcher exposes every GPU. An otherwise idle policy GPU
can therefore contain small contexts belonging to the other registered sims.
Never hide NVML processes, kill peers, or admit an unknown resource owner.
"""
import json
import os
from pathlib import Path
import subprocess
import time


def process(pid, proc_root=Path('/proc')):
    path = proc_root / str(pid)
    fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
    if fields[0] in ('Z', 'X'):
        raise ValueError('Peer process is not live')
    env = dict(item.split('=', 1) for item in (path / 'environ').read_bytes().decode().split('\0') if '=' in item)
    return dict(pid=pid, ppid=int(fields[1]), start_ticks=int(fields[19]),
                argv=(path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                environment={key: env.get(key) for key in ('EVAL_GPU', 'EVAL_SOURCE', 'CUDA_VISIBLE_DEVICES')})


def parse_compute(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].endswith(' MiB'):
            raise ValueError('Unparseable GPU ownership; do not assume idle')
        memory = fields[1][:-4]
        if not memory.isdigit():
            raise ValueError('Unknown GPU process memory')
        rows.append(dict(pid=int(fields[0]), memory_mib=int(memory)))
    if len({row['pid'] for row in rows}) != len(rows):
        raise ValueError('Duplicate GPU ownership row')
    return rows


def validate_owners(rows, candidates, gpu):
    if type(gpu) is not int or not 0 <= gpu < 8:
        raise ValueError('Explicit physical GPU required')
    for row in rows:
        peer = candidates.get(row['pid'])
        if (peer is None or peer['gpu'] == gpu or row['memory_mib'] > 512
                or row['memory_mib'] < 0):
            raise ValueError('Unknown, same-GPU, or substantial foreign GPU owner')
    if sum(row['memory_mib'] for row in rows) > 4096:
        raise ValueError('Peer contexts exceed admission reserve')
    return [dict(**row, peer=candidates[row['pid']]) for row in rows]


def check(parent, manifest_sha256, gpu):
    import continue_champion_final as c
    from common import SIM_PYTHON
    parent = Path(parent).resolve()
    if c.sha256(parent / 'manifest.json') != manifest_sha256:
        raise ValueError('Peer parent registration changed')
    m = c.read(parent / 'manifest.json'); _, original, _ = c.root_registration(parent)
    parts = c.checked_parts(parent, m, original); status = c.read(parent / 'status.json')
    candidates = {}
    for gs, worker_pid in status['worker_pids'].items():
        g = int(gs)
        if g == gpu:
            continue
        w = c.read(parent / 'workers' / f'gpu_{g}' / 'status.json')
        if w.get('status') != 'evaluating' or not w.get('part') or not w.get('sim_pid'):
            continue
        try:
            peer = process(w['sim_pid']); worker = process(worker_pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        pg, part = parts[w['part']]
        expected = [str(SIM_PYTHON), str(Path(m['runtime_source']) / 'scripts/eval/run_champion_task.py'),
                    '--task', part['task'], '--task-id', str(part['task_id']),
                    '--policy-config', original['configs'][g]['path'], '--output', str(parent / 'parts' / part['id']),
                    '--num-envs', str(part['num_envs']), '--indices', *map(str, part['indices'])]
        if (pg != g or w['pid'] != worker_pid or peer['ppid'] != worker_pid
                or w.get('runtime_commit') != original['source_commit'] or peer['argv'] != expected
                or peer['environment']['EVAL_GPU'] != str(g)
                or peer['environment']['EVAL_SOURCE'] != str(Path(m['runtime_source']) / 'scripts/eval')):
            raise ValueError('Registered peer process identity differs')
        candidates[peer['pid']] = dict(**peer, gpu=g, worker_pid=worker_pid,
                                      worker_start_ticks=worker['start_ticks'], part=part['id'])
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader'], text=True)
    rows = parse_compute(raw); owners = validate_owners(rows, candidates, gpu)
    free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    if not free.isdigit() or int(free) < 40 * 1024:
        raise ValueError('Insufficient free memory for unchanged model and simulator')
    for row in owners:
        again = process(row['pid'])
        if again != {key: row['peer'][key] for key in again}:
            raise ValueError('Peer changed during admission')
    return dict(schema='champion_registered_peer_gpu_admission_v1', observed_at=time.time(), gpu=gpu,
                parent_manifest=c.binding(parent / 'manifest.json'), nvml_compute=raw,
                owners=owners, free_memory_mib=int(free), process_pid=os.getpid(),
                unknown_owners_admitted=False, peers_stopped=False)

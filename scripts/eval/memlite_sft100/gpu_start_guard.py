"""Refuse occupied simulator GPUs except explicitly bound peer CUDA contexts.

Isaac creates small contexts on non-primary GPUs even with single-GPU rendering.
Those are not spare training processes, nor permission to ignore arbitrary PIDs.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def peer_identity(status_path, *, config_path, gpu, proc_root=Path('/proc')):
    path = Path(status_path).resolve()
    config_path = Path(config_path).resolve()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = json.loads(config_path.read_text())
    state = json.loads(path.read_text())
    case = config['cases'][state['case']]
    if (path.name != 'status.json' or state['status'] not in ('loading','running')
            or state['config_sha256'] != config_sha or case['task'] != state['task']
            or state['instance_id'] != config['instance_id'] or str(case['sim_gpu']) == str(gpu)):
        raise ValueError('Peer is not a live registered episode on another primary GPU')
    pid = state['pid']
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid peer PID')
    process = proc_root/str(pid)
    args = [x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
    if len(args) != 8 or Path(args[1]).name != 'run_causal_full_episode.py':
        raise ValueError('Peer PID is not the exact registered simulator command')
    options = dict(zip(args[2::2],args[3::2]))
    if (set(options) != {'--config','--case','--output'} or options['--case'] != state['case']
            or Path(options['--output']).resolve() != path.parent
            or hashlib.sha256(Path(options['--config']).read_bytes()).hexdigest() != config_sha):
        raise ValueError('Peer command and saved episode identity differ')
    # Read only the needed variable. Never print or save a process environment.
    env = dict(x.split(b'=',1) for x in (process/'environ').read_bytes().split(b'\0') if b'=' in x)
    if env.get(b'OMNIGIBSON_GPU_ID') != str(case['sim_gpu']).encode():
        raise ValueError('Peer primary GPU does not match its registered case')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(args[1]).parent,text=True).strip()
    if commit != state['source_commit']:
        raise ValueError('Peer is not running its recorded frozen source')
    return pid


def validate_snapshot(xml, allowed_pids):
    root = ET.fromstring(xml)
    cards = root.findall('gpu')
    if len(cards) != 1:
        raise ValueError('Expected exactly one queried simulator GPU')
    card = cards[0]
    def mib(text):
        if not text or not text.endswith(' MiB'):
            raise ValueError('GPU memory must be measurable, not N/A')
        return int(text[:-4])
    if mib(card.findtext('fb_memory_usage/free')) < 24576:
        raise ValueError('Insufficient free memory for a new full-resolution scene')
    processes = {}
    for item in card.findall('processes/process_info'):
        pid = int(item.findtext('pid'))
        memory = mib(item.findtext('used_memory'))
        if pid not in allowed_pids or not 0 <= memory <= 512:
            raise ValueError('Foreign process or nontrivial compute/render load on selected GPU')
        processes[pid] = max(processes.get(pid,0),memory)
    if sum(processes.values()) > 1024:
        raise ValueError('Too much auxiliary peer context memory')
    return dict(allowed_peer_pids=sorted(allowed_pids),observed_peer_mib=processes,
                free_mib=mib(card.findtext('fb_memory_usage/free')))


def check_simulator_gpu(gpu, *, peer_statuses=(), config_path):
    allowed = {peer_identity(path,config_path=config_path,gpu=gpu) for path in peer_statuses}
    xml = subprocess.check_output(['nvidia-smi','-i',str(gpu),'-q','-x'],text=True)
    return validate_snapshot(xml,allowed)

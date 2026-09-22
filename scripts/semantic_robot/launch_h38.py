"""One-shot H38 stages, after independent review; never retry a physical run.

The launcher lives in a separate immutable checkout. The experiment itself
always imports SOURCE, not this checkout. No live source/cache is overwritten.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import urllib.request


SOURCE = Path('/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_appearance_6f528b5')
COMMIT = '6f528b5e6d49950331e08f11217d7f5659038b45'
DIGEST = '11dfd7199b44bbed04a01deeb20962e4711ca05312e586e25125ee8ee419dfa6'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
ROOT = Path('/mnt/nvme_tmp/robodojo_agentic_20260921/h38_appearance')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_harness_runtime_20260921/gripper_v1_gates')
OLD_GATES = Path('/mnt/nvme_tmp/robodojo_agentic_20260921/gripper_v1_gates')
OLD_RUN = Path('/mnt/nvme_tmp/robodojo_agentic_20260921/h30_workspace_progress')
GPU_UUID = 'GPU-3e4fda8c-536e-5899-e877-b8be97032fe0'
PORT = 8930
CHILD_SOURCE = Path('/mnt/sdc1/robodojo/behavior_dev/git_worktrees/vlm_sft_h09z_evalwall_6f32887')
CHILD_COMMIT = '6f3288770f72b727a114d596cd5b98f577adc8c1'
CHILD_ROOT = Path('/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09y_grasp_only')
CHILD_GPU = 'GPU-c67cdb9d-ec23-7ca0-dce9-14a61c24c46b'
FLAGS = (
    'refine-grounding', 'visual-odometry', 'active-grasp-probe', 'grasp-motion',
    'robot-geometry-guards', 'approach-reorientation', 'approach-body-options',
    'approach-progress', 'held-object-inspection', 'persistent-grasp-tracks',
    'spatial-grasp-features', 'inspection-budget-aware', 'multicamera-inspection',
    'structured-planning', 'search-motion-recovery', 'odometry-self-exclusion',
    'gripper-completion-v1', 'odometry-match-refinement', 'workspace-posture',
    'near-contact-review', 'appearance-memory', 'carry-duration-v1',
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_review(review, launcher_sha):
    require(review.get('independent_review_passed') is True, 'Independent review required')
    require(review.get('source_commit') == COMMIT, 'Review/source mismatch')
    require(review.get('implementation_digest') == DIGEST, 'Review/public digest mismatch')
    require(review.get('launcher_sha256') == launcher_sha, 'Review/launcher mismatch')
    require(review.get('reviewer') == 'Astra-max-vlm_sft_resume_20260921', 'Independent reviewer identity')
    require(review.get('blocking_findings') == [], 'Unresolved review finding')


def check_child_binding(review, row):
    """Only an explicitly reviewed exact evaluation PID may have a tiny context."""
    allowed = {str(CHILD_ROOT / f'eval_t1_i{i}_{v}_v1')
               for i in (1, 71) for v in ('base', 'finetuned', 'proprio_history_nn')}
    require(review.get('reviewer') == 'Codex-parent', 'Exact parent context review required')
    require(type(review.get('pid')) is int and review['pid'] > 0
            and str(review['pid']) == row[1] and row[0] == GPU_UUID, 'Auxiliary PID mismatch')
    require(type(review.get('main_gpu')) is int and review['main_gpu'] == 3
            and type(review.get('max_auxiliary_MiB')) is int and review['max_auxiliary_MiB'] == 512
            and 0 < int(row[2]) <= 512, 'Auxiliary context size/GPU mismatch')
    require(review.get('source') == str(CHILD_SOURCE) and review.get('code_commit') == CHILD_COMMIT,
            'Unknown child source')
    require(review.get('output') in allowed, 'Unknown child evaluation slot')
    require(isinstance(review.get('launch_sha256'), str) and len(review['launch_sha256']) == 64,
            'Exact child launch hash required')


def check_child_process(review, row, all_apps):
    check_child_binding(review, row)
    proc = Path('/proc', row[1])
    require(proc.joinpath('cwd').resolve() == CHILD_SOURCE, 'Child cwd changed')
    actual = subprocess.check_output(['git', '-C', str(CHILD_SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == CHILD_COMMIT, 'Child source changed')
    launch_path = Path(review['output'] + '.launch.json')
    launch = read(launch_path)
    require(sha(launch_path) == review['launch_sha256'] and launch['pid'] == review['pid'],
            'Child launch/PID mismatch')
    arguments = proc.joinpath('cmdline').read_bytes().split(bytes([0]))
    require(arguments == [os.fsencode(part) for part in launch['command']] + [b''],
            'Child actual argv differs from reviewed launch')
    require(b'--output' in arguments and arguments[arguments.index(b'--output') + 1] == review['output'].encode(),
            'Child actual output mismatch')
    require(any(Path(os.fsdecode(arg)).name == 'native_eval_run.py' for arg in arguments if arg),
            'Not the registered evaluation runner')
    require(any(app[0] == CHILD_GPU and app[1] == row[1] for app in all_apps),
            'Child not actually present on main GPU3')
    return {'pid': review['pid'], 'main_gpu': 3, 'auxiliary_MiB': int(row[2]),
            'output': review['output'], 'launch_sha256': review['launch_sha256']}


def tree_bytes(base, cap_gib):
    """Count only this exact owned NVMe tree; do not follow links or skip errors."""
    queue, total = [base], 0
    device = Path('/mnt/nvme_tmp').stat().st_dev
    while queue:
        folder = queue.pop()
        s = folder.lstat()
        require(stat.S_ISDIR(s.st_mode) and s.st_dev == device
                and bool(s.st_mode & 0o444) and bool(s.st_mode & 0o111)
                and os.access(folder, os.R_OK | os.X_OK), f'Uncountable tree: {folder}')
        with os.scandir(folder) as entries:
            for entry in entries:
                s = entry.stat(follow_symlinks=False)
                require(s.st_dev == device, f'Unexpected mount: {entry.path}')
                if stat.S_ISDIR(s.st_mode):
                    queue.append(Path(entry.path))
                else:
                    require(stat.S_ISREG(s.st_mode) and os.access(entry.path, os.R_OK),
                            f'Unexpected file/link: {entry.path}')
                    total += s.st_size
    require(total < cap_gib * 1024**3, f'Tree cap: {base}')
    return total


def gate_review(name):
    launch = read(ROOT / f'launch_{name}.json')
    require(not Path('/proc', str(launch['pid'])).exists(), 'Prior gate still running')
    path = ROOT / f'gate_{name}_h38/result.json'
    result = read(path)
    require(result['gate_ok'] is True and result['gate_failures'] == []
            and result['implementation_digest'] == DIGEST, 'Exact gate failed')
    for key in ('gripper_completion_v1', 'odometry_match_refinement', 'workspace_posture',
                'near_contact_review', 'appearance_memory', 'carry_duration_v1'):
        require(result.get(key) is True, f'Gate flag mismatch: {key}')
    review = read(ROOT / f'gate_{name}_parent_review.json')
    require(review.get('reviewer') == 'Codex-parent'
            and review.get('full_numeric_and_RAW_review_passed') is True
            and review.get('video_review_passed') is True
            and review.get('result_sha256') == sha(path), 'Full gate review required')


def command_for(stage):
    if stage == 'model':
        output = ROOT / 'server_h38'
        command = ['/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python',
                   'scripts/semantic_robot/serve_v2.py', '--model',
                   '/mnt/sdc1/robodojo/behavior_dev/semantic_agent_v2_20260917/models/Qwen3.8-27B',
                   '--revision', REVISION, '--output', str(output), '--port', str(PORT),
                   '--max-calls', '431', '--structured-planning']
    else:
        policy = stage == 'policy'
        output = ROOT / ('radio_h38_fullstart' if policy else f'gate_{stage}_h38')
        command = ['/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python',
                   'scripts/semantic_robot/run_v2.py', '--output', str(output),
                   '--mode', 'agent' if policy else 'gate', '--harness', 'grounded',
                   '--task', '3' if stage == 'plates' else '0', '--gpu', '2', '--prefix', '0',
                   '--max-decisions', '192' if policy else '24',
                   '--max-controls', '6144' if policy else '1536',
                   '--max-seconds', '7200' if policy else '1200',
                   '--odometry-estimator', 'rgbd_joint', '--odometry-substep-controls', '6']
        command += ['--' + flag for flag in FLAGS]
        if policy:
            command += ['--budget-profile', 'fullstart192', '--uri', f'http://127.0.0.1:{PORT}',
                        '--expected-revision', REVISION]
            for name in ('radio', 'plates'):
                command += ['--gate-result', str(ROOT / f'gate_{name}_h38/result.json')]
    return output, command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('radio', 'plates', 'model', 'policy'))
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--child-context-review', type=Path,
                        help='Optional exact reviewed PID/launch binding, never a blanket auxiliary-process allowance')
    args = parser.parse_args()
    require(sha(args.review) == args.review_sha256, 'Review file hash changed')
    check_review(read(args.review), sha(Path(__file__)))
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == COMMIT, 'Wrong frozen experiment source')
    require(not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip(),
            'Dirty experiment source')
    paths = sorted((SOURCE / 'src/semantic_robot/v2').glob('*.py')) + [SOURCE / 'scripts/semantic_robot/run_v2.py']
    actual_digest = hashlib.sha256(b''.join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()
    require(actual_digest == DIGEST, 'Public implementation changed')
    for stage in ('policy', 'model'):
        require(not Path('/proc', str(read(OLD_RUN / f'launch_{stage}.json')['pid'])).exists(),
                'Old H30 process remains; preserve its evidence')
    require(shutil.disk_usage('/mnt/sdc1').free >= 32 * 1024**3, 'sdc1 reserve')
    require(shutil.disk_usage('/mnt/nvme_tmp').free >= 80 * 1024**3, 'NVMe reserve')
    if args.stage == 'plates':
        gate_review('radio')
    elif args.stage in ('model', 'policy'):
        for name in ('radio', 'plates'):
            gate_review(name)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu = next([s.strip() for s in line.split(',')] for line in rows if line.split(',')[0].strip() == '2')
    require(gpu[1] == GPU_UUID and int(gpu[3]) >= (20 if args.stage == 'policy' else 70) * 1024,
            'Exact GPU2/free memory')
    apps = [[s.strip() for s in line.split(',')] for line in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
        text=True).splitlines()]
    target = [row for row in apps if row[0] == GPU_UUID]
    own_model = read(ROOT / 'launch_model.json')['pid'] if args.stage == 'policy' else None
    auxiliary = []
    for row in target:
        if int(row[1]) == own_model:
            continue
        require(args.child_context_review is not None, 'Unregistered GPU2 process')
        auxiliary.append(check_child_process(read(args.child_context_review), row, apps))
    environment = read(OLD_GATES / 'launch_radio.json')['environment']
    environment['PYTHONPATH'] = str(SOURCE / 'src')
    for key, value in environment.items():
        if key not in ('PYTHONPATH', 'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE'):
            require(Path(value).resolve().is_relative_to(RUNTIME) and Path(value).is_dir(),
                    f'Exact owned runtime required: {key}')
    ROOT.mkdir(parents=True, exist_ok=True)
    sizes = {str(base): tree_bytes(base, cap) for base, cap in ((ROOT, 4), (RUNTIME, 16))}
    env = os.environ.copy()
    env.update(environment)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    health = None
    if args.stage == 'model':
        with socket.socket() as sock:
            require(sock.connect_ex(('127.0.0.1', PORT)) != 0, 'Port already occupied')
        env['CUDA_VISIBLE_DEVICES'] = '2'
        env['PYTHONPATH'] = ('/mnt/sdc1/robodojo/behavior_dev/semantic_structured_20260919/deps_817f944:'
                             '/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/deps:' + str(SOURCE / 'src'))
    elif args.stage == 'policy':
        require(any(int(row[1]) == own_model for row in target), 'Registered model not on GPU2')
        proc = Path('/proc', str(own_model))
        require(proc.joinpath('cwd').resolve() == SOURCE and str(ROOT / 'server_h38').encode()
                in proc.joinpath('cmdline').read_bytes().split(bytes([0])), 'Model process identity')
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=10) as response:
            health = json.load(response)
        expected = {'revision': REVISION, 'code_commit': COMMIT, 'calls': 0, 'visible_devices': '2',
                    'max_calls': 431, 'dtype': 'bfloat16', 'training_updates': 0, 'thinking': False,
                    'transformers': '5.7.0'}
        require(all(health.get(k) == v for k, v in expected.items()), 'Model health identity')
        require(health['structured_decoder']['commit'] == '817f944fcc8851917c40300a47f62d1defc3ffc3',
                'Structured decoder identity')
    output, command = command_for(args.stage)
    receipt = ROOT / f'launch_{args.stage}.json'
    require(not output.exists() and not receipt.exists(), 'Never reuse a submitted stage')
    for candidate in Path('/proc').iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            if candidate.stat().st_uid != os.geteuid():
                continue
            arguments = candidate.joinpath('cmdline').read_bytes().split(bytes([0]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(str(output).encode() not in arguments, f'Existing same-output process: {candidate.name}')
    with (ROOT / (output.name + '.log')).open('xb') as stream:
        proc = subprocess.Popen(command, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': proc.pid, 'submitted_bjt': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'code_commit': COMMIT, 'implementation_digest': DIGEST, 'source': str(SOURCE),
              'command': command, 'environment': {k: env[k] for k in environment},
              'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES', 'unset_for_simulator'),
              'gpu_before': gpu, 'tree_bytes_before': sizes, 'health_before': health,
              'allowed_auxiliary_contexts': auxiliary,
              'independent_review': str(args.review), 'independent_review_sha256': args.review_sha256,
              'launcher_sha256': sha(Path(__file__)), 'physical_new_resets': 0 if args.stage == 'model' else 1,
              'initialization_monitor_seconds': 900, 'max_MiB': 3072 if args.stage == 'policy' else 384}
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()

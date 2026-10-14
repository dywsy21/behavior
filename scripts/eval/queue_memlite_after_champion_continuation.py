"""Continue the same MEM remainder after the diagnosed champion continuation.

Separate queue/source, explicit old-queue lineage, no retries or policy changes.
Collect internal originals only; no public service, upload or submission.
"""
import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continue_champion_final as c
import package_champion_continuation as pack
import package_memlite_continuation as mem_pack
from queue_champion_after_memlite import dependency_ready


def validate_order(mem, champion, authorization, mem_source, champion_source):
    mem, mm = c.shared.registration(mem, coordinator_source=mem_source)
    champion, cm = c.registration(champion, coordinator_source=champion_source)
    root, original, pin = c.root_registration(champion)
    c.shared.root_registration(mem)
    authorization = Path(authorization).resolve(); auth = c.read(authorization)
    seal = c.read(mm['seal']['path'])
    if (auth.get('schema') != 'user_authorized_eval_priority_stop_v1'
            or mm['prior'] != auth['mem_job'] or auth['champion_job'] != str(root)
            or auth['mem_manifest_sha256'] != mm['prior_manifest']['sha256']
            or auth['champion_manifest_sha256'] != pin['sha256']
            or seal.get('priority_authorization') != c.binding(authorization)
            or mm.get('automatic_retries') is not False or mm.get('policy_changed') is not False
            or mm.get('optimizer_updates') != 0 or original.get('automatic_retries') is not False):
        raise ValueError('Exact original user priority and unchanged MEM remainder required')
    return mem, mm, champion, cm, authorization, root


def replacement_queue(mem, champion_root, previous, name, authorization, champion=None, pending_mem_upgrade=None):
    if not re.fullmatch(r'queue-[A-Za-z0-9_-]+', name):
        raise ValueError('Queue basename required')
    previous, queue = Path(previous).resolve(), mem / name
    if previous == queue or queue.exists() or queue.is_symlink():
        raise ValueError('Preserve the old queue; use a new directory inside the same MEM run')
    if pending_mem_upgrade is not None:
        from pending_mem_upgrade import verify
        verify(pending_mem_upgrade,previous,mem,champion,authorization)
        return queue,c.binding(previous/'status.json')
    if previous.parent != mem:
        raise ValueError('A different pending MEM requires explicit audited supersession')
    p = previous / 'status.json'; old = c.read(p)
    if (old.get('status') != 'needs_diagnosis' or not old.get('finished') or old.get('phases') != []
            or old.get('mem_job') != str(mem)):
        raise ValueError('Only a dead never-launched champion-failure queue may be replaced')
    c.require_exited([old['pid']])
    if old.get('error') == "RuntimeError('Champion needs diagnosis; do not launch MEM or retry champion')":
        if old['champion_job'] != str(champion_root):
            raise ValueError('Previous queue used another champion root')
        expected = [c.binding(mem / 'manifest.json'), c.binding(champion_root / 'manifest.json'),
                    c.binding(authorization), c.read(mem / 'manifest.json')['seal']]
    elif old.get('error') == "RuntimeError('Champion continuation needs diagnosis; do not start MEM or retry')":
        old_champion = Path(old['champion_job']).resolve()
        if (champion is None or c.read(Path(champion)/'manifest.json')['prior'] != str(old_champion)
                or old['champion_root'] != str(champion_root)
                or c.root_registration(old_champion)[0] != champion_root):
            raise ValueError('Replacement must follow this exact stopped champion continuation')
        ancestor = old['bindings'][-1]; c.check_binding(ancestor)
        if Path(ancestor['path']).parent.parent != mem:
            raise ValueError('Foreign queue ancestry')
        c.require_exited([c.read(ancestor['path'])['pid']])
        expected = [c.binding(mem/'manifest.json'), c.binding(old_champion/'manifest.json'),
                    c.binding(authorization), c.read(mem/'manifest.json')['seal'],
                    c.read(old_champion/'manifest.json')['seal'], c.binding(champion_root/'manifest.json'), ancestor]
    else:
        raise ValueError('Unclassified previous queue failure')
    if old.get('bindings') != expected:
        raise ValueError('Previous queue was bound to another run or authorization')
    if (mem / 'status.json').exists() or (mem / 'STOP').exists():
        raise ValueError('Never launch a previously attempted MEM continuation')
    return queue, c.binding(p)


def mem_command(mem, mem_source):
    return [sys.executable, str(Path(mem_source) / 'scripts/eval/continue_memlite_final.py'),
            'supervise', '--job', str(mem)]


def run(mem, champion, authorization, mem_source, champion_source, previous_queue, queue_name, output_root,
        *, pending_mem_upgrade=None):
    mem_source, champion_source = Path(mem_source).resolve(), Path(champion_source).resolve()
    mem, mm, champion, cm, auth, root = validate_order(mem, champion, authorization, mem_source, champion_source)
    queue, old = replacement_queue(mem, root, previous_queue, queue_name, auth, champion,pending_mem_upgrade)
    output_root = Path(output_root).resolve()
    if (not output_root.is_dir() or any((output_root / name).exists() for name in ('champion-results', 'mem-results'))
            or any(output_root.is_relative_to(p) for p in (mem, champion, root))):
        raise ValueError('Fresh result destinations outside original jobs required')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.REPO, text=True).strip()
    c.frozen_source(c.REPO, commit)
    owner = (mem / 'queue-owner.lock').open('a'); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Recheck after acquiring ownership, before creating or launching anything.
    replacement_queue(mem, root, previous_queue, queue_name, auth, champion,pending_mem_upgrade)
    queue.mkdir()
    bindings = [c.binding(mem / 'manifest.json'), c.binding(champion / 'manifest.json'), c.binding(auth),
                mm['seal'], cm['seal'], c.binding(root / 'manifest.json'), old]
    if pending_mem_upgrade is not None:
        bindings.insert(-1,c.binding(pending_mem_upgrade))
    state = dict(status='waiting_for_champion_continuation', pid=os.getpid(), started=time.time(), phases=[],
        mem_job=str(mem), champion_job=str(champion), champion_root=str(root), bindings=bindings,
        mem_source=str(mem_source), champion_source=str(champion_source), queue_source=str(c.REPO),
        queue_source_commit=commit, automatic_retries=False, output_root=str(output_root))
    if pending_mem_upgrade is not None:
        state['pending_mem_upgrade']=c.binding(pending_mem_upgrade)
    def publish():
        state['updated'] = time.time(); c.atomic_json(queue / 'status.json', state)
    def unchanged():
        for row in bindings:
            c.check_binding(row)
        if (queue / 'STOP').exists() or (mem / 'STOP').exists():
            raise InterruptedError('Requested queue stop')
    try:
        publish()
        while True:
            unchanged(); status = c.read(champion / 'status.json'); summary = c.read(champion / 'summary.json')
            ready = pack.terminal_ready(status, summary)
            alive = bool(status.get('pid') and Path(f'/proc/{status["pid"]}').exists())
            if ready and not alive:
                break
            if not ready and not alive:
                raise RuntimeError('Champion disappeared without final accounted evidence')
            time.sleep(30)
        state['status'] = 'auditing_and_collecting_champion'; publish()
        state['champion_collection'] = pack.package(champion, bindings[1]['sha256'], champion_source,
                                                    output_root / 'champion-results')
        publish(); unchanged()
        validate_order(mem, champion, auth, mem_source, champion_source)
        if (mem / 'status.json').exists():
            raise ValueError('MEM was started by another owner')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPUs still owned; never displace another process')
        argv = mem_command(mem, mem_source)
        phase = dict(name='remaining_mem', status='running', started=time.time(), argv=argv)
        state['status'] = 'running_mem_remainder'; state['phases'].append(phase); publish()
        with (queue / 'mem-supervisor.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=mem_source, env=os.environ | dict(PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            phase['pid'] = child.pid; publish(); result = child.wait()
        phase.update(returncode=result, finished=time.time(), status='completed' if result == 0 else 'failed'); publish()
        if result or not dependency_ready(c.read(mem / 'status.json'), c.read(mem / 'summary.json'), continuation=True):
            raise RuntimeError('MEM needs diagnosis; no retry')
        unchanged(); state['status'] = 'auditing_and_collecting_mem'; publish()
        state['mem_collection'] = mem_pack.package(mem, bindings[0]['sha256'], output_root / 'mem-results')
        state['status'] = 'both_internal_evaluations_finished_with_disclosed_errors_not_submitted'
    except BaseException as error:
        state.update(status='needs_diagnosis', error=repr(error)); raise
    finally:
        state['finished'] = time.time(); publish(); owner.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('mem-job', 'champion-job', 'authorization', 'mem-source', 'champion-source',
                 'previous-queue', 'output-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--queue-name', required=True)
    p.add_argument('--pending-mem-upgrade',type=Path)
    a = p.parse_args()
    run(a.mem_job, a.champion_job, a.authorization, a.mem_source, a.champion_source,
        a.previous_queue, a.queue_name, a.output_root,pending_mem_upgrade=a.pending_mem_upgrade)

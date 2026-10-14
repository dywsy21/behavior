"""Replace a never-started MEM queue with the same cases and frozen actor.

Only the waiting queue is stopped. The running champion is never signalled.
All prior receipts, queue states and the old unused MEM registration are kept.
"""
import argparse
from pathlib import Path
import time

import continue_memlite_final as m

IDENTITY=('prior','prior_manifest','seal','root_manifest','runtime_source','runtime_commit',
          'config_path','config_sha256','pack','pack_manifest_sha256','tasks','cases',
          'inherited_completed','initial_execution_errors','administrative_restarts',
          'expected','automatic_retries','policy_changed','optimizer_updates')


def same_remainder(old,new):
    if any(old.get(k)!=new.get(k) for k in IDENTITY):
        raise ValueError('Pending upgrade changed policy, data, or original case accounting')
    def schedule(value):
        return [(g['gpu'],p['task'],p['task_id'],p['num_envs'],i)
                for g in value['groups'] for p in g['parts'] for i in p['indices']]
    if schedule(old)!=schedule(new):
        raise ValueError('Pending upgrade regrouped/reordered original physical slots')
    if new.get('continue_on_batch_errors') is not True or new.get('isolate_physical_batches') is not True:
        raise ValueError('Replacement must have registered batch/worker isolation')


def unused(job):
    if (job/'status.json').exists() or (job/'STOP').exists() or any((job/'parts').iterdir()) or any((job/'workers').iterdir()):
        raise ValueError('MEM already attempted; never replace or reroll it as a pending job')


def request(old_queue,new_mem,champion,authorization,old_source,new_source,output):
    old_queue,new_mem,champion,authorization,old_source,new_source,output=map(
        lambda p:Path(p).resolve(),(old_queue,new_mem,champion,authorization,old_source,new_source,output))
    old_mem=old_queue.parent
    if output.exists() or not output.parent.is_dir() or (old_queue/'STOP').exists():
        raise ValueError('Fresh explicit waiting-queue supersession required')
    _,old=m.registration(old_mem,coordinator_source=old_source)
    _,new=m.registration(new_mem,coordinator_source=new_source)
    same_remainder(old,new);unused(old_mem);unused(new_mem)
    before=m.read(old_queue/'status.json')
    if (before.get('status')!='waiting_for_champion_continuation' or before.get('phases')!=[]
            or before.get('mem_job')!=str(old_mem) or before.get('champion_job')!=str(champion)
            or not Path('/proc/'+str(before['pid'])).exists()
            or m.read(champion/'status.json')['status']!='running'):
        raise ValueError('Only the live, still-waiting queue may be superseded')
    pins=[m.binding(old_mem/'manifest.json'),m.binding(new_mem/'manifest.json'),
          m.binding(champion/'manifest.json'),m.binding(authorization)]
    if not all(pin in before['bindings'] for pin in (pins[0],pins[2],pins[3])):
        raise ValueError('Waiting queue identity/priority differs')
    row=dict(schema='pending_mem_isolation_upgrade_v1',created=time.time(),old_queue=str(old_queue),
        old_mem=str(old_mem),new_mem=str(new_mem),champion=str(champion),authorization=str(authorization),
        old_source=str(old_source),new_source=str(new_source),bindings=pins,before=before,
        instruction='User requested continuous evaluation without a single-worker global stop',
        champion_stop_requested=False,policy_changed=False,scored_cases_replayed=0)
    m.atomic_json(output,row)
    (old_queue/'STOP').touch(exist_ok=False)
    return row


def verify(receipt,old_queue,new_mem,champion,authorization):
    receipt,old_queue,new_mem,champion,authorization=map(lambda p:Path(p).resolve(),
        (receipt,old_queue,new_mem,champion,authorization))
    row=m.read(receipt);before=row['before'];old_mem=old_queue.parent
    if (row.get('schema')!='pending_mem_isolation_upgrade_v1'
            or row['old_queue']!=str(old_queue) or row['new_mem']!=str(new_mem)
            or row['old_mem']!=str(old_mem) or row['champion']!=str(champion)
            or row['authorization']!=str(authorization) or row['policy_changed'] is not False
            or row['champion_stop_requested'] is not False or row['scored_cases_replayed']!=0
            or before['status']!='waiting_for_champion_continuation' or before['phases']!=[]):
        raise ValueError('Wrong pending-MEM supersession')
    expected=[m.binding(old_mem/'manifest.json'),m.binding(new_mem/'manifest.json'),
              m.binding(champion/'manifest.json'),m.binding(authorization)]
    if row['bindings']!=expected:
        raise ValueError('Changed pending-MEM bindings')
    _,old=m.registration(old_mem,coordinator_source=row['old_source'])
    _,new=m.registration(new_mem,coordinator_source=row['new_source'])
    same_remainder(old,new);unused(old_mem);unused(new_mem)
    stopped=m.read(old_queue/'status.json')
    if (stopped['status']!='needs_diagnosis' or stopped.get('phases')!=[]
            or stopped.get('error')!="InterruptedError('Requested queue stop')"
            or not before['started']<=row['created']<=(old_queue/'STOP').stat().st_mtime<=stopped['finished']
            or any(stopped.get(k)!=before.get(k) for k in ('pid','mem_job','champion_job','bindings'))):
        raise ValueError('Waiting queue was not safely superseded before MEM execution')
    m.require_exited([stopped['pid']])
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old-queue','new-mem','champion','authorization','old-source','new-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    request(a.old_queue,a.new_mem,a.champion,a.authorization,a.old_source,a.new_source,a.output)
    print('Requested stop of the waiting queue only; champion evaluation remains running.')

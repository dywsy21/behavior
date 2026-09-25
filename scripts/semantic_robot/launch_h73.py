"""One synchronous-I/O native gate, retaining all original motion gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from semantic_robot.v2.synchronous_io import validate_installed, reference_seconds, VIEWS
import launch_h69 as gate

ORIGINAL_IDENTITY, ORIGINAL_VALIDATE = gate.identity, gate.validate_result
ORIGINAL_FLAGS = gate.FLAGS


def identity(base):
    validate_installed()
    return ORIGINAL_IDENTITY(base)


def validate_result(result, digest):
    ORIGINAL_VALIDATE(result, digest)
    if result.get('synchronous_io_v1') is not True:
        raise ValueError('Explicit synchronous I/O required')
    raw=(gate.ROOT/'gate/native_io.jsonl').read_bytes()
    rows=[json.loads(x) for x in raw.splitlines()]
    control=[r for r in rows if r['kind']=='control']
    capture=[r for r in rows if r['kind']=='capture']
    expected={'controls':len(control),'attempts':len(control),'captures':len(capture),
              'journal_sha256':hashlib.sha256(raw).hexdigest()}
    if (result.get('native_io') != expected or len(control)!=result['controls'] or
            not capture or rows[0].get('kind')!='initialize' or rows[-1].get('kind')!='close' or
            sum(r.get('kind')=='initialize' for r in rows)!=1 or
            sum(r.get('kind')=='close' for r in rows)!=1 or
            any(r.get('completed') is not True for r in rows)):
        raise ValueError('Incomplete native I/O journal')
    cursor=rows[0]['before']
    for row in rows:
        if row.get('kind') not in ('initialize','control','capture','close') or row['before']!=cursor:
            raise ValueError('Unaccounted clock advancement between I/O transactions')
        if row['kind']!='control' and row['after']!=cursor:
            raise ValueError('Non-control transaction advanced physics')
        cursor=row['after']
    for index,row in enumerate(control,1):
        if (row['call']!=index or row['actual_render_on_step'] is not False or
                row['after']['physics_index']-row['before']['physics_index']!=4 or
                abs(row['after']['simulation_time']-row['before']['simulation_time']-1/30)>1e-7):
            raise ValueError('Control-to-physics clock mismatch')
    for index,row in enumerate(capture,1):
        if (row['capture']!=index or row['after']!=row['before'] or
                set(row['references'])!=VIEWS or
                any(abs(reference_seconds(v)-row['before']['simulation_time'])>1e-6 for v in row['references'].values())):
            raise ValueError('Stale or physically advancing camera capture')
    sensors=json.loads((gate.ROOT/'gate/sensor_checks.json').read_text())
    if len(sensors)!=len(capture): raise ValueError('Capture receipts do not bind runner observations')
    for check,clock in zip(sensors,capture):
        if set(check['cameras'])!=VIEWS: raise ValueError('Missing camera receipt')
        for camera in check['cameras'].values():
            t=camera.get('native_time',{})
            if (camera.get('freshness_proof')!='native_reference_time' or
                    t.get('reference_time_verified') is not True or
                    t.get('physics_index')!=clock['before']['physics_index'] or
                    t.get('simulation_time')!=clock['before']['simulation_time'] or
                    t.get('physics_ticks_in_capture')!=0):
                raise ValueError('Runner used an unverified capture')


def configure():
    gate.ROOT=Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h73_synchronous_io_v1')
    gate.RUNTIME=Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h73_synchronous_io_v1')
    gate.WALL_SECONDS=1200
    gate.FLAGS=ORIGINAL_FLAGS+('synchronous-io-v1',)
    gate.identity,gate.validate_result=identity,validate_result
    base=gate.configure();base.ENTRYPOINT=Path(__file__).resolve()
    return base


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',action='store_true');mode.add_argument('--supervise',action='store_true')
    args=parser.parse_args();base=configure()
    (gate.launch if args.launch else gate.supervise)(base)

"""One synchronous-I/O native gate, retaining all original motion gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from semantic_robot.v2.synchronous_io import validate_installed, VIEWS
from semantic_robot.v2.render_batch import validate_batch, validate_advance, make_receipts
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
    reads=[r for r in rows if r['kind']=='read']
    primes=[r for r in rows if r['kind']=='prime']
    expected={'controls':len(control),'attempts':len(control),'captures':len(capture),
              'reads':len(reads),'primes':len(primes),
              'journal_sha256':hashlib.sha256(raw).hexdigest()}
    if (result.get('native_io') != expected or len(control)!=result['controls'] or
            not capture or len(reads)!=len(capture) or len(primes)!=1 or
            rows[0].get('kind')!='initialize' or rows[1].get('kind')!='prime' or rows[-1].get('kind')!='close' or
            sum(r.get('kind')=='initialize' for r in rows)!=1 or
            sum(r.get('kind')=='close' for r in rows)!=1 or
            any(r.get('completed') is not True for r in rows)):
        raise ValueError('Incomplete native I/O journal')
    cursor=rows[0]['before']
    pending=None
    for row in rows:
        kind=row.get('kind')
        if kind not in ('initialize','control','prime','capture','read','close') or row['before']!=cursor:
            raise ValueError('Unaccounted clock advancement between I/O transactions')
        if row['kind']!='control' and row['after']!=cursor:
            raise ValueError('Non-control transaction advanced physics')
        cursor=row['after']
        if pending is not None and kind!='read':
            raise ValueError('Captured pixels must be verified before another transaction')
        if kind=='capture': pending=row
        if kind=='read':
            if (pending is None or row['capture']!=pending['capture'] or row['batch']!=pending['batch']):
                raise ValueError('Torn or unmatched RGB-D read')
            pending=None
    if pending is not None: raise ValueError('Unfinished RGB-D read')
    if primes[0].get('discarded') is not True: raise ValueError('Prime must not become an actor observation')
    validate_batch(primes[0]['batch'])
    for index,row in enumerate(control,1):
        if (row['call']!=index or row['actual_render_on_step'] is not False or
                row['after']['physics_index']-row['before']['physics_index']!=4 or
                abs(row['after']['simulation_time']-row['before']['simulation_time']-1/30)>1e-7):
            raise ValueError('Control-to-physics clock mismatch')
    for index,row in enumerate(capture,1):
        if row['capture']!=index or row['after']!=row['before']:
            raise ValueError('Stale or physically advancing camera capture')
        validate_advance(row['batch'],row['baseline'])
        validate_advance(row['batch'],capture[index-2]['batch'] if index>1 else primes[0]['batch'])
    sensors=json.loads((gate.ROOT/'gate/sensor_checks.json').read_text())
    if len(sensors)!=len(capture): raise ValueError('Capture receipts do not bind runner observations')
    for check,clock,read in zip(sensors,capture,reads):
        if set(check['cameras'])!=VIEWS: raise ValueError('Missing camera receipt')
        times=make_receipts(clock['batch'],clock['before'],clock['capture'])
        for view,camera in check['cameras'].items():
            t=camera.get('native_time',{})
            if (camera.get('freshness_proof')!='native_render_batch_v2' or t!=times[view] or
                    read.get('buffer_hashes',{}).get(view)!={k:camera.get(k) for k in ('rgb_sha256','depth_sha256')}):
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

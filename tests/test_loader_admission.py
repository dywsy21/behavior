import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
import loader_admission as m


def test_two_slots_are_fifo_and_later_request_cannot_overtake(tmp_path):
    requests = [m.enqueue(tmp_path, f'worker{i}') for i in range(4)]
    assert m.try_admit(tmp_path, requests[1]) is None
    a = m.try_admit(tmp_path, requests[0]); b = m.try_admit(tmp_path, requests[1])
    assert a is not None and b is not None
    assert m.try_admit(tmp_path, requests[2]) is None
    a.close()
    assert m.try_admit(tmp_path, requests[3]) is None
    c = m.try_admit(tmp_path, requests[2]); assert c is not None
    b.close(); d = m.try_admit(tmp_path, requests[3]); assert d is not None
    c.close(); d.close(); d.close()
    assert [m.read(p)['status'] for p in requests] == ['released'] * 4
    assert len(list((tmp_path / 'loader-admission').glob('request-*.json'))) == 4


def test_dead_or_reused_pid_cannot_block_the_queue(tmp_path):
    stale = m.enqueue(tmp_path, 'old'); fresh = m.enqueue(tmp_path, 'new')
    value = m.read(stale); value['process_start_ticks'] = 'not-this-process'; m.write(stale, value)
    slot = m.try_admit(tmp_path, fresh)
    assert slot is not None and m.read(stale)['status'] == 'abandoned'
    slot.close()


def test_waiting_past_thirty_minutes_is_not_a_model_failure(tmp_path, monkeypatch):
    a = m.try_admit(tmp_path, m.enqueue(tmp_path, 'a'))
    b = m.try_admit(tmp_path, m.enqueue(tmp_path, 'b'))
    clock = [1000.]; waits = []
    monkeypatch.setattr(m.time, 'time', lambda: clock[0])
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    def wait(seconds):
        waits.append(seconds); clock[0] += 1900
        if len(waits) == 2: a.close()
    monkeypatch.setattr(m.time, 'sleep', wait)
    c = m.loader_lock(tmp_path, purpose='never-started-gpu')
    assert waits == [2, 2]
    receipt = m.read(c.request)
    assert receipt['admitted_at'] - receipt['queued_at'] == 3800
    c.close(); b.close()


def test_global_stop_cancels_waiting_ticket_without_touching_active_slots(tmp_path):
    a = m.try_admit(tmp_path, m.enqueue(tmp_path, 'a'))
    (tmp_path / 'STOP').touch()
    with pytest.raises(InterruptedError, match='Administrative stop'):
        m.loader_lock(tmp_path, purpose='waiting')
    states = [m.read(p)['status'] for p in sorted((tmp_path / 'loader-admission').glob('request-*.json'))]
    assert states == ['admitted', 'cancelled']
    a.close()


def test_live_owner_identity_and_foreign_request_are_rejected(tmp_path):
    request = m.enqueue(tmp_path, 'owned')
    value = m.read(request); value['pid'] = -1; m.write(request, value)
    with pytest.raises(ValueError): m.try_admit(tmp_path, request)
    foreign = tmp_path / 'request.json'; foreign.write_text(json.dumps(value))
    with pytest.raises(ValueError): m.try_admit(tmp_path, foreign)
    assert m.process_stamp(os.getpid()) is not None and m.process_stamp(987654321) is None


def test_actual_child_processes_share_two_fifo_slots(tmp_path):
    import subprocess
    script=('import sys,time;sys.path.insert(0,sys.argv[1]);import loader_admission as m;'
            'from pathlib import Path;slot=m.loader_lock(Path(sys.argv[2]),purpose="child");'
            'time.sleep(.05);slot.close()')
    children=[]
    try:
        for _ in range(6):
            children.append(subprocess.Popen([sys.executable,'-c',script,str(Path(m.__file__).parent),str(tmp_path)],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        for child in children:
            out,error=child.communicate(timeout=20)
            assert child.returncode==0,(out,error)
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait()
    rows=[m.read(p) for p in (tmp_path/'loader-admission').glob('request-*.json')]
    assert len(rows)==6 and len({r['pid'] for r in rows})==6
    assert all(r['status']=='released' for r in rows)
    assert [r['sequence'] for r in sorted(rows,key=lambda r:r['admitted_at'])]==list(range(6))
    live=0
    for _,change in sorted([(r['admitted_at'],1) for r in rows]+[(r['released_at'],-1) for r in rows]):
        live+=change;assert 0<=live<=2
    assert live==0

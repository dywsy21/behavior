"""FIFO admission for expensive model/scene loading, not an execution timeout.

Only two existing loader_N.lock slots may be held. Waiting workers cannot be
overtaken by a worker starting its next task. Actual model/scene watchdogs stay
in the caller and begin after admission. Queue records are retained as evidence.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_stamp(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


@contextmanager
def guard(job):
    queue = Path(job) / 'loader-admission'; queue.mkdir(exist_ok=True)
    with (queue / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield queue


def enqueue(job, purpose):
    with guard(job) as queue:
        counter = queue / 'counter.json'
        sequence = read(counter)['sequence'] + 1 if counter.exists() else 0
        write(counter, dict(sequence=sequence))
        request = queue / f'request-{sequence:08d}-{os.getpid()}.json'
        if request.exists():
            raise ValueError('Never reuse a loader ticket')
        write(request, dict(status='waiting', sequence=sequence, pid=os.getpid(),
            process_start_ticks=process_stamp(os.getpid()), purpose=purpose, queued_at=time.time()))
        return request


class Admission:
    def __init__(self, job, request, stream):
        self.job, self.request, self.stream = job, request, stream

    def close(self):
        if self.stream is None:
            return
        try:
            with guard(self.job):
                value = read(self.request)
                write(self.request, value | dict(status='released', released_at=time.time()))
        finally:
            self.stream.close(); self.stream = None


def try_admit(job, request):
    with guard(job) as queue:
        if request.parent != queue:
            raise ValueError('Foreign loader ticket')
        own = read(request)
        if (own['pid'] != os.getpid() or own['process_start_ticks'] != process_stamp(os.getpid())
                or own['status'] != 'waiting'):
            raise ValueError('Loader ticket is not owned by this waiting process')
        if (Path(job) / 'STOP').exists():
            raise InterruptedError('Administrative stop')
        waiting = []
        for path in sorted(queue.glob('request-*.json')):
            value = read(path)
            if value['status'] != 'waiting':
                continue
            stamp = process_stamp(value['pid'])
            if stamp is None or stamp != value['process_start_ticks']:
                write(path, value | dict(status='abandoned', observed_dead_at=time.time()))
            else:
                waiting.append((value['sequence'], path))
        if not waiting or min(waiting)[1] != request:
            return None
        for index in range(2):
            stream = (Path(job) / f'loader_{index}.lock').open('a')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close(); continue
            try:
                write(request, own | dict(status='admitted', slot=index, admitted_at=time.time()))
            except BaseException:
                stream.close(); raise
            return Admission(job, request, stream)
        return None


def loader_lock(job, *, purpose='loading'):
    request = enqueue(job, purpose)
    try:
        while True:
            admission = try_admit(job, request)
            if admission is not None:
                return admission
            # No 30-minute queue deadline: being queued is not a hung model.
            # The original post-admission watchdogs are unchanged in worker().
            time.sleep(2)
    except BaseException:
        with guard(job):
            value = read(request)
            if value['status'] == 'waiting':
                write(request, value | dict(status='cancelled', cancelled_at=time.time()))
        raise

"""Read-only GPU telemetry. A sampled peak is not a 24GB hardware acceptance."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def process_memory_mib(text, pid):
    values = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError('Unexpected nvidia-smi process fields')
        if int(row[0].strip()) != pid:
            continue
        value = float(row[1].strip())
        if not math.isfinite(value) or value < 0:
            raise ValueError('Unavailable process memory')
        values.append(value)
    if len(values) > 1:
        raise ValueError('Duplicate PID for the registered single GPU')
    # Before CUDA initializes, no process row is an actual zero, not a query error.
    return values[0] if values else 0.0


class ProcessMemoryProfile:
    def __init__(self, output, gpu, *, interval=0.5):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Positive sampling interval required')
        self.output, self.gpu = Path(output), str(gpu)
        self.interval, self.pid = interval, os.getpid()
        self.phase = 'loading'
        self.stop_event = threading.Event()
        self.thread = None
        self.rows = []
        self.jax_rows = []
        self.device = None

    def _query(self, fields):
        return subprocess.check_output([
            'nvidia-smi', '-i', self.gpu, fields, '--format=csv,noheader,nounits',
        ], text=True, timeout=10).strip()

    def sample(self):
        row = dict(time=time.time(), pid=self.pid, gpu=self.gpu, phase=self.phase)
        try:
            text = self._query('--query-compute-apps=pid,used_gpu_memory')
            row['process_mib'] = process_memory_mib(text, self.pid)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            row['error'] = repr(error)
        self.rows.append(row)
        with (self.output/'gpu-memory-samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def start(self):
        if self.thread is not None or (self.output/'gpu-memory-samples.jsonl').exists():
            raise ValueError('Fresh one-use memory profile required')
        fields = list(csv.reader([self._query('--query-gpu=uuid,name,memory.total')]))[0]
        if len(fields) != 3 or float(fields[2]) <= 0:
            raise ValueError('Single identified GPU required')
        self.device = dict(uuid=fields[0].strip(), name=fields[1].strip(), total_mib=float(fields[2]))
        self.sample()
        def monitor():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=monitor, name='champion-memory-profile', daemon=True)
        self.thread.start()

    def record_jax(self, label, stats):
        # The caller gets device.memory_stats only after its synchronous model call.
        if stats is None:
            self.jax_rows.append(dict(label=label, error='memory_stats unavailable'))
            return
        self.jax_rows.append(dict(label=label, time=time.time(), stats={str(k): int(v) for k,v in stats.items()}))

    def finish(self):
        if self.thread is None:
            raise ValueError('Memory profile was not started')
        self.stop_event.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError('Memory sampler did not stop')
        self.sample()
        valid = [row for row in self.rows if 'process_mib' in row]
        errors = [row for row in self.rows if 'error' in row]
        phases = sorted({row['phase'] for row in valid})
        return dict(
            schema='champion_gpu_memory_profile_v1', device=self.device, pid=self.pid,
            sampling_interval_seconds=self.interval, samples=len(self.rows), sample_errors=len(errors),
            sampled_process_peak_mib=max((row['process_mib'] for row in valid), default=None),
            phase_peak_mib={phase:max(row['process_mib'] for row in valid if row['phase']==phase) for phase in phases},
            sampled_peak_is_lower_bound=True, jax_samples=self.jax_rows,
            not_24gb_hardware_acceptance=True, not_container_acceptance=True,
            not_formal_ema_acceptance=True,
        )

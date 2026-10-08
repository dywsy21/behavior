"""Bounded eight-GPU SUM correctness and bandwidth; no models or simulation."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--port', type=int, default=29881)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--nccl-library', type=Path)
    args = parser.parse_args()
    if not 120 <= args.timeout <= 900:
        raise ValueError('Probe timeout must be between 120 and 900 seconds')
    if args.rank is None:
        library = None
        if args.nccl_library:
            resolved = args.nccl_library.resolve(strict=True)
            with resolved.open('rb') as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
            library = dict(path=str(resolved), sha256=digest.hexdigest())
        if args.out.exists():
            raise ValueError('Fresh probe directory required')
        args.out.mkdir(parents=True)
        processes, locks = [], []
        try:
            import fcntl
            from bootstrap import bootstrap
            bootstrap()
            from gpu_admission import check_gpu_available
            for rank in range(8):
                lock = open(f'/run/ti/behavior_stage3_20260930/gpu_{rank}.lock', 'a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(lock)
                check_gpu_available(rank)
            for rank in range(8):
                with (args.out/f'rank{rank}.log').open('w') as log:
                    processes.append(subprocess.Popen([sys.executable, '-u', __file__, '--out', str(args.out),
                        '--rank', str(rank), '--port', str(args.port), '--timeout', str(args.timeout)],
                        env=os.environ | dict(CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='1',
                                              NCCL_DEBUG=os.environ.get('NCCL_DEBUG', 'WARN'),
                                              NCCL_DEBUG_FILE=str(args.out/f'nccl{rank}.log'),
                                              TORCH_NCCL_ASYNC_ERROR_HANDLING='1') |
                            ({'LD_PRELOAD': library['path']} if library else {}),
                        stdout=log, stderr=subprocess.STDOUT))
            deadline = time.time() + args.timeout
            for process in processes:
                if process.wait(timeout=max(1., deadline-time.time())):
                    raise RuntimeError('NCCL rank failed; inspect per-rank log')
            rows = [json.loads((args.out/f'rank{rank}.json').read_text()) for rank in range(8)]
            result = dict(passed=True, ranks=rows,
                          communication_library=library,
                          communication_env={key: os.environ[key] for key in
                              ('NCCL_CUMEM_HOST_ENABLE', 'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE',
                               'NCCL_SHM_DISABLE', 'NCCL_SOCKET_IFNAME') if key in os.environ},
                          max_32MiB_seconds=max(r['32'][0] for r in rows),
                          estimated_actor_reduce_seconds=max(r['32'][0] for r in rows) * 2420.843853 / 32,
                          estimate_not_end_to_end=True)
            (args.out/'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            for lock in locks:
                lock.close()
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    print('PHASE import_torch', time.time(), flush=True)
    import torch
    import torch.distributed as dist
    from synchronous import Collective
    print('PHASE cuda_warmup', time.time(), 'architectures', torch.cuda.get_arch_list(), flush=True)
    torch.cuda.set_device(0)
    warmup = torch.ones(1024, device='cuda').sum().item()
    if warmup != 1024:
        raise RuntimeError('CUDA arithmetic failed before communication')
    print('PHASE collective_initialize', time.time(), flush=True)
    print('LOADED_NCCL', sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                               if '/libnccl.so' in line}), flush=True)
    Collective.initialize(args.rank, 8, f'tcp://127.0.0.1:{args.port}', timeout=args.timeout-30)
    print('PHASE collective_ready', time.time(), flush=True)
    result = {}
    for size in (32, 256):
        samples = []
        values = torch.empty(size * 2**20 // 4, device='cuda', dtype=torch.float32)
        for step in range(5):
            values.fill_(args.rank + 1)
            dist.barrier()
            torch.cuda.synchronize()
            started = time.monotonic()
            dist.all_reduce(values)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            if not bool((values == 36).all()):
                raise RuntimeError('Incorrect eight-rank SUM')
            if step:
                samples.append(elapsed)
        result[str(size)] = [sum(samples)/len(samples), samples]
        print('PHASE sum_passed', size, result[str(size)], flush=True)
    (args.out/f'rank{args.rank}.json').write_text(json.dumps(result))
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()

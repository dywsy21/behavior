"""Bounded eight-GPU SUM correctness and bandwidth; no models or simulation."""
import argparse
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
    args = parser.parse_args()
    if args.rank is None:
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
                    processes.append(subprocess.Popen([sys.executable, __file__, '--out', str(args.out),
                        '--rank', str(rank), '--port', str(args.port)],
                        env=os.environ | dict(CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='1',
                                              NCCL_DEBUG=os.environ.get('NCCL_DEBUG', 'WARN'),
                                              TORCH_NCCL_ASYNC_ERROR_HANDLING='1'),
                        stdout=log, stderr=subprocess.STDOUT))
            deadline = time.time() + 180
            for process in processes:
                if process.wait(timeout=max(1., deadline-time.time())):
                    raise RuntimeError('NCCL rank failed; inspect per-rank log')
            rows = [json.loads((args.out/f'rank{rank}.json').read_text()) for rank in range(8)]
            result = dict(passed=True, ranks=rows,
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
    import torch
    import torch.distributed as dist
    from synchronous import Collective
    Collective.initialize(args.rank, 8, f'tcp://127.0.0.1:{args.port}', timeout=90)
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
    (args.out/f'rank{args.rank}.json').write_text(json.dumps(result))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

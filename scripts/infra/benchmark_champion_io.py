"""CPU-only random native V3 decode/transport measurement, not a training run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--batches', type=int, default=6)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        raise ValueError('Explicit CPU-only environment and fresh result required')
    if not 1 <= args.workers <= 32 or not 4 <= args.batches <= 10:
        raise ValueError('Short bounded I/O diagnostic required')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from behavior_champion.dataset import NativeV3Dataset, collate_native, worker_init
    torch.set_num_threads(1)
    data = NativeV3Dataset(args.release, 'train')
    # Full global sample shuffle, not convenient neighbouring frames/first episodes.
    indices = np.random.default_rng(17).permutation(len(data))[:args.batches*256]
    loader = DataLoader(Subset(data, indices), batch_size=32, num_workers=args.workers,
                        shuffle=False, collate_fn=collate_native, worker_init_fn=worker_init,
                        multiprocessing_context='spawn', prefetch_factor=1, persistent_workers=False)
    iterator = iter(loader)
    rows = []
    identities = []
    for step in range(args.batches):
        started = time.monotonic()
        chunks = [next(iterator) for _ in range(8)]
        # Measure actual parent-process receive/collation of full native RGB.
        batch = {key: np.concatenate([chunk[key] for chunk in chunks])
                 for key in chunks[0] if key != 'identity'}
        source = [r for chunk in chunks for r in chunk['identity']]
        if len(source) != 256 or len({r['task'] for r in source}) < 2:
            raise ValueError('Batch must contain 256 real mixed-task observations')
        if any(r['split'] != 'train' or r['frame'] % 16 != r['phase'] for r in source):
            raise ValueError('Split/stride changed')
        row = dict(batch=step, seconds=time.monotonic()-started,
                   distinct_tasks=len({r['task'] for r in source}),
                   bytes=sum(x.nbytes for x in batch.values()), warmup=step < 2)
        rows.append(row); identities.extend(source)
        print(json.dumps(row), flush=True)
        del batch, chunks
    try: next(iterator)
    except StopIteration: pass
    else: raise ValueError('Unexpected extra data')
    timed_seconds = sum(r['seconds'] for r in rows[2:])
    result = dict(status='cpu_native_decode_and_transport_completed', workers=args.workers,
                  real_samples=len(identities), total_train_stride16_samples=len(data),
                  episodes=len(data.episodes), seconds_per_batch_median=float(np.median([r['seconds'] for r in rows[2:]])),
                  timed_seconds=timed_seconds, seconds_per_batch_aggregate=timed_seconds/(len(rows)-2),
                  examples_per_second_aggregate=(len(rows)-2)*256/timed_seconds,
                  rows=rows, sampled_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  identities=identities, no_normalization_resize_tokenization=True,
                  caveat='Raw I/O only. Includes CPU decode and process transport; excludes champion transforms, accelerator compute, checkpoints and eval. Not an end-to-end ETA.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('identities','rows')}), flush=True)


if __name__ == '__main__': main()

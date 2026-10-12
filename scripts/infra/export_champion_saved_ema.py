"""CPU-only export of a pinned durable formal checkpoint while training runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reserve-gib', type=int, default=775)
    args = ap.parse_args()
    if (os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or args.reserve_gib < 256 or args.output.exists()
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()):
        raise ValueError('CPU-only, fresh output, clean frozen code and >=256GiB reserve required')
    root = args.root.resolve(); started = time.monotonic()
    if not args.checkpoint.resolve().is_relative_to(root) or not args.output.resolve().is_relative_to(root):
        raise ValueError('Use explicit checkpoint and export paths inside the registered shared root')
    from behavior_champion.inference import checked_sources
    from behavior_champion.checkpoint_export import read_checkpoint_ema, verify_parent_assets
    from behavior_champion.export import export_ema
    reference = root / 'src/champion2026-native-v1'; openpi = root / 'src/openpi2026-native-v1'
    checked_sources(reference, openpi, REPO)
    sys.path[:0] = [str(reference / 'src'), str(openpi / 'src')]
    receipts = root / 'runs/champion_assets_20261011'
    layout = json.loads((receipts / 'task-layout-admitted-v2.json').read_text())['layout']
    state, receipt, contract = read_checkpoint_ema(args.checkpoint,
        manifest_sha256=args.manifest_sha256, layout=layout)
    assets = verify_parent_assets(root / 'models/champion2025/shared50-11490241',
        receipts / 'shared50-verified-v1.json', expected_sha256=contract['parent_receipt_sha256'])
    exported = export_ema(args.output, state, receipt=receipt, assets=assets,
        layout=layout, contract=contract, reserve_bytes=args.reserve_gib << 30)
    result = dict(status='durable_ema_exported_and_author_roundtrip_verified', export=exported,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        seconds=time.monotonic()-started, optimizer_updates=0, gpu_devices=0,
        training_source_or_state_modified=False, physical_success_measurement=False)
    with (args.output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

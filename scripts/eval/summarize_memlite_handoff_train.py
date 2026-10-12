"""Merge all fixed TRAIN shards before inspecting paired field/choice metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_calibration_launch import read, write_new
from recovery_corpus import file_sha
from recovery_generation_shards import merge_generation_shards
from recovery_handoff_diagnosis_summary import paired_diagnostics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control-shards',type=Path,nargs='+',required=True)
    ap.add_argument('--candidate-shards',type=Path,nargs='+',required=True)
    ap.add_argument('--pack-manifest',type=Path,required=True)
    ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if args.output.exists() or file_sha(args.pack_manifest) != args.manifest_sha256:
        raise ValueError('Fresh output and original transport manifest required')
    pack = read(args.pack_manifest); merged = {}; evidence = {}
    for name, paths in (('control',args.control_shards),('candidate',args.candidate_shards)):
        parts=[]; evidence[name]=[]
        for path in paths:
            parts.append((read(path/'manifest.json'),read(path/'result.json'),
                [json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]))
            evidence[name].append(dict(path=str(path),sha256={f:file_sha(path/f) for f in
                                                           ('manifest.json','result.json','rows.jsonl')}))
        receipt, rows, seconds = merge_generation_shards(parts)
        model = pack['models'][name]
        if (len(rows) != 383 or receipt['data_split'] != 'train' or receipt['optimizer_updates'] != 0
                or receipt['checkpoint_sha256'] != model['original']['checkpoint']['sha256']
                or receipt['model_export_sha256'] != model['sha256']
                or receipt['portable_manifest_sha256'] != args.manifest_sha256
                or receipt['schedule_sha256'] != pack['common']['schedule_sha256']
                or receipt['runtime_node'] != 'RTX10383' or not receipt['target_free']):
            raise ValueError('Foreign, partial or changed TRAIN diagnostic model')
        merged[name]=(receipt,rows,seconds)
    # Reject cross-hardware/source comparisons, even if identities match.
    if any(merged['control'][0][k] != merged['candidate'][0][k] for k in
           ('source_commit','config_sha256','runtime_node','portable_manifest_sha256')):
        raise ValueError('Paired diagnostic changed code, config, hardware or inputs')
    summary = paired_diagnostics(merged['control'][1],merged['candidate'][1])
    summary.update(status='completed_paired_train_diagnosis',evidence=evidence,
        pack_manifest_sha256=args.manifest_sha256,
        analysis_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    args.output.mkdir(parents=True)
    for name,(receipt,rows,seconds) in merged.items():
        out=args.output/name;out.mkdir()
        write_new(out/'manifest.json',receipt)
        write_new(out/'result.json',dict(receipt,status='completed',rows=len(rows),shard_seconds=seconds))
        with (out/'rows.jsonl').open('x') as stream:
            for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    write_new(args.output/'result.json',summary)
    print(json.dumps(summary,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()

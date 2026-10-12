"""Execute the same fixed TRAIN diagnosis on a verified, self-contained pack."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/eval'),
               str(REPO/'scripts/infra'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import file_sha, digest
from recovery_generation_shards import generation_shard
from recovery_handoff_inference import tree_sha
from recovery_handoff_portable import verify_pack, materialize_arch, check_prefix_rows
from recovery_process_cache import isolated_cache_env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registration',type=Path,required=True)
    ap.add_argument('--pack',type=Path,required=True)
    ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--model',choices=('control','candidate'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--shard-index',type=int,default=0)
    ap.add_argument('--cpu-only',action='store_true')
    args = ap.parse_args(); registration = read(args.registration)
    target_root = Path(registration['target_root']).resolve()
    if (not args.pack.resolve().is_relative_to(target_root) or not args.output.resolve().is_relative_to(target_root)
            or Path(sys.executable).absolute() != Path(registration['target_python'])
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
            or args.output.exists()):
        raise ValueError('Registered RTX paths/interpreter and clean frozen source required')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    def idle():
        if gpu not in tuple(map(str,range(1,8))) or subprocess.check_output(
                ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('Use one actually idle registered GPU; never stop other jobs')
    if args.cpu_only:
        if gpu != '': raise ValueError('CPU acceptance must hide CUDA')
    else: idle()
    manifest = verify_pack(args.pack,args.manifest_sha256,registration,file_sha(args.registration))
    accepted = read(args.pack/'cpu-prefix-audit.json')
    model_spec = manifest['models'][args.model]
    checkpoint = bound(args.pack,dict(path=model_spec['path'],sha256=model_spec['sha256']))
    args.output.mkdir(parents=True); os.environ.update(isolated_cache_env(args.output)); started = time.monotonic()
    import torch
    torch.set_num_threads(2); torch.manual_seed(17)
    original = torch.load(args.pack/'inputs.pt',map_location='cpu',weights_only=False)
    if len(original) != 383 or tree_sha(original) != manifest['input_tree_sha256']:
        raise ValueError('Portable input tensor values changed')
    check_prefix_rows([row[-1] for row in original],accepted)
    schedule = [(row[1],row[2]) for row in original]
    local, sharding = generation_shard(schedule,args.shard_index,args.num_shards)
    if digest(schedule) != manifest['common']['schedule_sha256']:
        raise ValueError('Different TRAIN order after transport')
    selected = []
    for position,split,index in local:
        row = original[position]
        if tuple(row[:3]) != (position,split,index) or row[3]['original_identity']['split'] != 'train':
            raise ValueError('Foreign split, position or source identity')
        selected.append(row)
    config = dict(arch=materialize_arch(read(args.pack/'arch.json'),args.pack.resolve()))
    from check_memlite_periodic_processor import make_token_processor
    from diagnose_memlite_handoff_train import choice_layout, run_diagnostic
    from g05.models.g05.g05_policy_memlite_planner_outcome import G05PolicyMEMLitePlannerOutcome as Policy
    from types import SimpleNamespace
    tokens = make_token_processor(config); arch=config['arch']; head=arch['planner_outcome']
    shell = SimpleNamespace(processor=tokens,training=False,planner_only=True,
        max_chunk_token_length=arch['max_chunk_token_length'],max_pad_token_length=arch.get('max_pad_token_length'),
        supervise_task_parent_format=head.get('supervise_task_parent_format',False),
        task_parent_formats=head.get('task_parent_format_by_task',{}))
    shell._context_positions = Policy._context_positions
    shell._mask_unsupervised_ar_fields = lambda *args: Policy._mask_unsupervised_ar_fields(shell,*args)
    for _position,_split,_index,_request,_target,prefix,prepared,record in selected:
        encoded = tokens.encode_inference([prefix],device=torch.device('cpu'),mode='ar',training=False)
        if (tree_sha(encoded) != record['encoded_sha256'] or tree_sha(prepared['pixel_values']) != record['pixels_sha256']
                or digest(choice_layout(shell,prepared)) != record['choice_layout_sha256']):
            raise ValueError('RTX tokenizer, pixels or first-choice mask differs from A800 CPU acceptance')
    common = dict(manifest['common'],source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        checkpoint_sha256=model_spec['original']['checkpoint']['sha256'],
        fit_result_sha256=model_spec['original']['fit_result']['sha256'],
        portable_manifest_sha256=args.manifest_sha256,model_export_sha256=model_spec['sha256'],
        runtime_node='RTX10383',sharding=sharding,schedule_sha256=digest(schedule),
        checkpoint_tensor_bytes_unchanged=True,model_comparison_same_hardware=True)
    write_new(args.output/'cpu-prefix-audit.json',dict(common,status='passed',rows=[row[-1] for row in selected]))
    if args.cpu_only:
        write_new(args.output/'result.json',dict(common,status='completed_cpu_only',rows=len(selected),
                  model_forwards=0,seconds=time.monotonic()-started))
        return
    del original,tokens,shell
    import gc
    gc.collect(); idle()
    if torch.__version__ != '2.7.1+cu128': raise ValueError('Registered RTX torch build changed')
    import transformers
    if transformers.__version__ != '4.57.1': raise ValueError('Registered tokenizer library changed')
    run_diagnostic(args,checkpoint,config,selected,common,started)


if __name__=='__main__': main()

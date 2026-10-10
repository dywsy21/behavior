"""Read-only matched FM losses on every already-approved TRAIN/DEV action.

This tests learning/phase coverage, not task success. No new data, optimizer,
checkpoint selection, augmentation or CAL/test access is performed.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts'), str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import file_sha, digest
from recovery_sft_data import VerifiedRecoveryActionDataset
from recovery_action_diagnostic import summarize_action_losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); root = Path(cfg['root'])
    if (cfg['schema'] != 'recovery_matched_action_phase_diagnostic_v1' or a.output.exists()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Explicit pinned read-only recipe, clean source and new output required')
    if (len(cfg['models']) != 3 or [m['name'] for m in cfg['models']] != ['parent','control','later-actions']
            or len(cfg['noise_seeds']) != len(set(cfg['noise_seeds']))):
        raise ValueError('Fixed parent and both terminal SFT arms with unique matched noise seeds required')
    for model in cfg['models']:
        if file_sha(root/model['path']) != model['sha256']:
            raise ValueError('Changed immutable model')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str, range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('One actually idle A800 required')
    import torch
    from g05.data.memlite_stage1_dataset import collate_stage1, to_device
    from g05.utils.training.stage1_model import configuration, restore_model, make_processor
    from g05.utils.training.stage1_runtime import atomic_json
    from g05.models.g05.qwen35 import vision
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2); vision._flash_attn_varlen=None; vision._flash_attn_backend=None
    names=json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    config=configuration(root,'low',names)
    if file_sha(config['stats_path']) != cfg['stats_sha256']:
        raise ValueError('Different normalized objective')
    corpus=root/cfg['corpus']; inventory=json.loads((corpus/'audit/inventory.json').read_text())
    datasets=[VerifiedRecoveryActionDataset(corpus/'admission',corpus/'raw',inventory,config,
                  split=split,admission_sha256=cfg['admission_sha256']) for split in ('train','dev')]
    samples=[]; metadata=[]
    try:
        for ds in datasets:
            # Diagnostic preprocessing is explicitly deterministic evaluation,
            # including TRAIN anchors; no random crop can imitate learning.
            ds.processor=make_processor(config,False)
            for i,item in enumerate(ds.rows):
                row=item['candidate']; bundle=row['actor_input']['issued_skills_semantic_json']
                if isinstance(bundle,str): bundle=json.loads(bundle)
                if len(bundle)!=1: raise ValueError('This diagnostic admits one known skill per anchor')
                with torch.random.fork_rng(devices=[0]): samples.append(ds[i])
                metadata.append(dict(sample_id=row['sample_id'],source_group=row['source_group'],
                    event_id=item['approval']['event_id'],split=row['split'],control_step=row['control_step'],
                    mechanism=bundle[0]['verb']))
    finally:
        for ds in datasets: ds.reader.close()
    if {split:sum(r['split']==split for r in metadata) for split in ('train','dev')} != cfg['expected_rows']:
        raise ValueError('Approved phase set changed')
    a.output.mkdir(parents=True)
    status=dict(status='loading',pid=os.getpid(),config_sha256=file_sha(a.config),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        admission_sha256=cfg['admission_sha256'],rows_sha256=digest(metadata),
        optimizer_updates=0,whole_task_sr=False,policy_promotion_authorized=False)
    atomic_json(a.output/'status.json',status); atomic_json(a.output/'anchors.json',metadata)
    results=[]
    for bound in cfg['models']:
        saved=torch.load(root/bound['path'],map_location='cpu',mmap=True,weights_only=False)
        model,restored=restore_model(config,'low',state=saved['model_state_dict']);del saved;gc.collect()
        model.configure_recovery_expert_only()
        if {k:len(v) for k,v in model.coordination_trainable_parameter_groups().items()} != {'action_expert':322}:
            raise ValueError('Changed objective boundary')
        model.eval().cuda()
        before={str(frozen):parameter_digest(model,frozen=frozen) for frozen in (True,False)}
        measurements=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for sample,meta in zip(samples,metadata):
                batch=to_device(collate_stage1([sample]),torch.device('cuda'))
                for seed in cfg['noise_seeds']:
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(seed+int(meta['sample_id'][:8],16))
                        loss,metrics=model(batch)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite FM diagnostic')
                    measurements.append(dict(meta,noise_seed=seed,
                        numerator=float(metrics['row_loss_numerator'].sum()),
                        denominator=float(metrics['row_loss_denominator'].sum())))
                    model.model.ar_helper._last_ce_cache=None
                del batch
        after={str(frozen):parameter_digest(model,frozen=frozen) for frozen in (True,False)}
        if before != after or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Read-only objective changed policy or created gradients')
        atomic_json(a.output/(bound['name']+'-rows.json'),measurements)
        result=dict(model=bound,summary=summarize_action_losses(measurements,
            [r['sample_id'] for r in metadata],cfg['noise_seeds']),fingerprints_before=before,
            fingerprints_after=after,actual_forwards=len(measurements),restoration=restored,
            rows_sha256=file_sha(a.output/(bound['name']+'-rows.json')))
        results.append(result);atomic_json(a.output/'status.json',dict(status,status='measuring',completed_models=results))
        del model;gc.collect();torch.cuda.empty_cache()
    status.update(status='matched_action_phase_loss_diagnostic_completed',models=results,
        independent_events_not_inflated_by_extra_anchors=True,calibration_or_test_used=False)
    atomic_json(a.output/'result.json',status);atomic_json(a.output/'status.json',status)


if __name__=='__main__': main()

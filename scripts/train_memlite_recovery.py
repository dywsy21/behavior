"""Finite H1/L0 derivative SFT, separate from immutable Stage-1 training.

No automatic launch permission. Admissions and an explicit pinned run ticket
are checked before CUDA. Engineering mode uses only original expert data and
is capped at TWO cumulative updates; no recovery labels or claimed effects.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_train_contract import validate_launch
from recovery_corpus import file_sha


def parameter_digest(model,*,frozen):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad == frozen: continue
        h.update(name.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticket',type=Path,required=True)
    p.add_argument('--component',choices=('H1','L0'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engineering',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args();ticket,recipe=validate_launch(a.ticket,a.component,engineering=a.engineering)
    if 'RECOVERY_DEADLINE' not in os.environ: raise RuntimeError('Use the cumulative-budget recovery launcher')
    deadline=float(os.environ['RECOVERY_DEADLINE']);stop=Path(os.environ['RECOVERY_STOP_FILE'])
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(): raise ValueError('Use frozen clean source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if commit!=ticket['source_commit']: raise ValueError('Ticket code differs from actual import source')
    if recipe['preferred_node'] not in subprocess.check_output(['hostname','-I'],text=True).split(): raise ValueError('Wrong node')
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from g05.data.memlite_stage1_dataset import Stage1Dataset,collate_stage1,to_device
    from g05.utils.training.stage1_model import configuration,restore_model
    from g05.utils.training.stage1_runtime import atomic_json,capture_rng,restore_rng,save_checkpoint,load_checkpoint,init_wandb
    from g05.utils.training.recovery_runtime import shard_finite_batch,global_objective_step,schedule_fingerprint,collective_error
    from g05.models.g05.qwen35 import vision
    from recovery_sft_data import VerifiedRecoveryActionDataset,finite_mixture_schedule
    from recovery_planner_data import VerifiedRecoveryPlannerDataset
    from recovery_features import balanced_event_schedule
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=ticket['world_size'] or rank!=local: raise ValueError('One registered node only')
    torch.set_num_threads(2);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=900))
    torch.manual_seed(recipe['seed']+rank);random.seed(recipe['seed']+rank);np.random.seed(recipe['seed']+rank)
    root=Path(recipe['root']);release=root/recipe['expert_release'];branch='high' if a.component=='H1' else 'low'
    manifest=json.loads((release/'manifest.json').read_text())
    config=configuration(root,branch,manifest['task_names']);parent=recipe['parents'][branch]
    config.update(initial_weights=str(root/parent['path']),initial_weights_sha256=parent['sha256'])
    expected_stats=(recipe['stats_sha256'] if branch=='low' else ticket['files']['high_stats']['sha256'])
    if file_sha(config['stats_path'])!=expected_stats: raise ValueError('Normalizer mismatch')
    accepted=json.loads((release/'acceptance.json').read_text())
    expert_index=json.loads(Path(ticket['files']['expert_index']['path']).read_text())
    if (accepted['status']!='ACCEPTED' or not all(accepted['gates'].values())
            or file_sha(release/'manifest.json')!=accepted['manifest_sha256']
            or expert_index['manifest_sha256']!=accepted['manifest_sha256']): raise ValueError('Unaccepted original rehearsal source')
    experts=Stage1Dataset(release,config,branch,'train')
    original_eval=Stage1Dataset(release,config,branch,'eval')
    by_task={k:[r['candidate'] for r in v] for k,v in expert_index['rows'].items()}
    if a.engineering:
        flat=[by_task[k][0] for k in sorted(by_task,key=int)[:32]]
        schedule=[dict(event_pass=0,rows=[('expert',i) for i in flat[:16]]),
                  dict(event_pass=0,rows=[('expert',i) for i in flat[16:32]])]
        new=dev=None
        data_sha=accepted['manifest_sha256']
    else:
        inventory=json.loads(Path(ticket['files']['inventory']['path']).read_text())
        kwargs=dict(release=ticket['admission'],raw_root=ticket['raw_root'],inventory=inventory,config=config,
                    admission_sha256=ticket['files']['admission']['sha256'])
        if a.component=='L0':
            new=VerifiedRecoveryActionDataset(**kwargs,split='train');dev=VerifiedRecoveryActionDataset(**kwargs,split='dev')
            schedule=list(finite_mixture_schedule(new.rows,by_task,batch_size=ticket['global_batch'],
                maximum_event_passes=ticket['event_passes'],seed=recipe['seed'],
                allow_extended_event_fit=recipe.get('extended_event_fit',False)))
        else:
            histories=[json.loads(x) for x in Path(ticket['files']['history']['path']).read_text().splitlines()]
            feedback=[json.loads(x) for x in Path(ticket['files']['feedback']['path']).read_text().splitlines()]
            feedback_receipt=json.loads(Path(ticket['files']['feedback_receipt']['path']).read_text())
            if (feedback_receipt['feedback_sha256']!=ticket['files']['feedback']['sha256']
                    or feedback_receipt['high_sha256']!=parent['sha256'] or feedback_receipt['diagnostic_only']):
                raise ValueError('Invalid OOF feedback release')
            kwargs.update(history=histories,feedback=feedback,evidence_root=ticket['evidence_root'],high_sha256=parent['sha256'])
            new=VerifiedRecoveryPlannerDataset(**kwargs,split='train');dev=VerifiedRecoveryPlannerDataset(**kwargs,split='dev')
            # Recovery-only RETRY targets would teach a degenerate planner.
            # Rehearse normal original plans as well, exactly as L0 retains
            # original actions; never mislabel these as new recovery events.
            schedule=list(finite_mixture_schedule(new.rows,by_task,batch_size=ticket['global_batch'],
                maximum_event_passes=ticket['event_passes'],seed=recipe['seed'],pool='planner',
                allow_extended_event_fit=recipe.get('extended_event_fit',False)))
        data_sha=ticket['files']['admission']['sha256']
    schedule=schedule[:ticket['maximum_updates']]
    binding=dict(component=a.component,source_commit=commit,parent_sha256=parent['sha256'],admission_sha256=data_sha,
        recipe_sha256=file_sha(a.ticket),world_size=world,micro_batch=ticket['micro_batch'])
    fingerprint=schedule_fingerprint(schedule,binding=binding)
    if rank==0:
        if not a.resume: a.output.mkdir(parents=True,exist_ok=False)
        atomic_json(a.output/'schedule.json',dict(fingerprint=fingerprint,binding=binding,schedule=schedule))
    dist.barrier()
    saved=receipt=None
    if a.resume: saved,receipt=load_checkpoint(a.output/'checkpoints',fingerprint)
    # One rank verifies the large immutable parent, then all load the exact
    # explicitly provided state. Never use old B1500/A42500 implicit init.
    error=None
    if rank==0:
        try:
            if file_sha(root/parent['path'])!=parent['sha256']: raise ValueError('Wrong final SFT parent')
        except Exception as exc: error=exc
    collective_error(error,device=device,phase='parent verification')
    parent_data=torch.load(root/parent['path'],map_location='cpu',mmap=True,weights_only=False) if saved is None else saved
    model,restoration=restore_model(config,branch,state=parent_data['model_state_dict'])
    if branch=='low': model.configure_recovery_expert_only()
    groups={k:len(v) for k,v in model.coordination_trainable_parameter_groups().items()}
    wanted={'action_expert':322} if branch=='low' else {'planner_vlm':326}
    if groups!=wanted: raise ValueError('Wrong derivative gradient boundary')
    del parent_data;gc.collect()
    model.cuda().train()
    frozen_before=parameter_digest(model,frozen=True) if rank==0 else None
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=recipe[a.component]['learning_rate'],weight_decay=recipe[a.component].get('weight_decay',.01))
    state=dict(step=0,save_sequence=0,fingerprint=fingerprint,consumed_seconds=0.,run_id=uuid.uuid4().hex[:12])
    if saved is not None:
        state=dict(saved['state']);optimizer.load_state_dict(saved['optimizer_state_dict'])
        restore_rng(saved['rng_by_rank'][rank]);del saved;gc.collect()
    box=[state['run_id'] if rank==0 else None];dist.broadcast_object_list(box,0);state['run_id']=box[0]
    wb=None;error=None
    if rank==0:
        try:
            cfg=dict(recipe['wandb'],name=a.output.name,group=recipe['wandb']['group']+('-engineering' if a.engineering else ''))
            wb=init_wandb(cfg,a.output,run_id=state['run_id'],resume=a.resume,
                metadata=dict(component=a.component,engineering_only=a.engineering,parent_sha256=parent['sha256'],
                    data_sha256=data_sha,source_commit=commit,schedule_fingerprint=fingerprint,trainable_groups=groups))
        except Exception as exc: error=exc
    collective_error(error,device=device,phase='W&B online initialization')
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    def load(rows):
        # Stage1Dataset's deterministic CPU transforms call manual_seed.
        # In its original worker processes that cannot affect the trainer's
        # CUDA generator; here synchronous IO must preserve it explicitly.
        with torch.random.fork_rng(devices=[local]):
            samples=[experts[i] if origin=='expert' else new[i] for origin,i in rows]
        return to_device(collate_stage1(samples),device)
    def objective(net,batch):
        loss,metrics=net(batch)
        model.model.ar_helper._last_ce_cache=None
        return loss,metrics['row_loss_denominator'].sum()
    def save():
        rngs=[None]*world;dist.all_gather_object(rngs,capture_rng());error=None
        state['save_sequence']+=1
        state['consumed_seconds']=float(os.environ['RECOVERY_PREVIOUS_SECONDS'])+time.monotonic()-float(os.environ['RECOVERY_STARTED'])
        if rank==0:
            try:
                save_checkpoint(a.output/'checkpoints',model=model,optimizer=optimizer,state=dict(state),rng_by_rank=rngs)
            except Exception as exc: error=exc
        collective_error(error,device=device,phase='atomic checkpoint');dist.barrier()
    def evaluate():
        evaluation={};rng=capture_rng();was_training=model.training;model.eval()
        try:
            fixed_eval=np.load(release/'fixed_eval_indices.npy').reshape(100,32)[:,0].tolist()
            dev_indices={}
            if dev is not None:
                for i,item in enumerate(dev.rows):
                    dev_indices.setdefault((item['candidate']['source_group'],item['approval']['event_id']),i)
            for scope,ds,indices in [('original_heldout',original_eval,fixed_eval)]+(
                    [('verified_dev',dev,list(dev_indices.values()))] if dev is not None else []):
                stats=torch.zeros(2,dtype=torch.float64,device=device);error=None
                torch.manual_seed(9183+rank)
                try:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        for i in indices[rank::world]:
                            with torch.random.fork_rng(devices=[local]): sample=ds[i]
                            loss,m=model(to_device(collate_stage1([sample]),device))
                            if not torch.isfinite(loss): raise ValueError('Nonfinite heldout objective')
                            stats+=torch.stack([m['row_loss_numerator'].sum(),m['row_loss_denominator'].sum()]).double()
                            model.model.ar_helper._last_ce_cache=None
                except Exception as exc: error=exc
                collective_error(error,device=device,phase='heldout '+scope)
                dist.all_reduce(stats)
                evaluation[scope]=dict(loss=float(stats[0]/stats[1]),denominator=float(stats[1]),rows=len(indices))
        finally:
            restore_rng(rng);model.train(was_training)
        if rank==0:
            with (a.output/'evaluations.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=state['step'],evaluation=evaluation))+'\n')
            wb.log({'train/update':state['step'],**{'eval/'+k+'/loss':v['loss'] for k,v in evaluation.items()}})
        return evaluation
    # Pin the parent comparison BEFORE any derivative optimizer step, using
    # exactly the same heldout examples/noise seeds as the final evaluation.
    if state['step']==0 and not a.engineering: evaluate()
    evaluation_every=recipe.get('evaluation_every_updates',0)
    if type(evaluation_every) is not int or evaluation_every<0: raise ValueError('Invalid evaluation interval')
    checkpoint_every=ticket.get('checkpoint_every_updates',1)
    if not isinstance(checkpoint_every,int) or checkpoint_every<1: raise ValueError('Invalid save interval')
    target=min(len(schedule),a.stop_after or len(schedule))
    last_saved=state['step'] if a.resume else -1
    if a.stop_after is not None and not 1<=a.stop_after<=ticket['maximum_updates']: raise ValueError('Invalid cumulative stop step')
    while state['step']<target:
        flag=torch.tensor(int(stopping[0] or stop.exists() or time.monotonic()+180>=deadline),device=device)
        dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item(): break
        batch=schedule[state['step']];micros=shard_finite_batch(batch['rows'],rank=rank,world_size=world,micro_batch=ticket['micro_batch'])
        before=time.monotonic()
        metrics=global_objective_step(ddp,optimizer,micros,load_micro=load,objective=objective,device=device,
            autocast_factory=lambda:torch.autocast('cuda',dtype=torch.bfloat16))
        state['step']+=1
        model.model.ar_helper._last_ce_cache=None
        metrics.update(update=state['step'],seconds=time.monotonic()-before,event_pass=batch['event_pass'])
        if rank==0:
            with (a.output/'updates.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
            wb.log({'train/update':state['step'],**{'train/'+k:v for k,v in metrics.items()},'scope/engineering_only':int(a.engineering)})
        if state['step']%checkpoint_every==0 or state['step']==target:
            save();last_saved=state['step']
        if evaluation_every and state['step']%evaluation_every==0 and state['step']<target:
            evaluate()
    if state['step']!=last_saved: save()
    # Read-only heldout objectives, separated from the original rehearsal.
    # Use one representative per accepted event, not long-clip frame inflation.
    evaluation=evaluate()
    error=None
    if rank==0:
        try:
            frozen_after=parameter_digest(model,frozen=True)
            if frozen_before!=frozen_after: raise ValueError('Frozen model changed')
            result=dict(status='completed_finite_schedule' if state['step']==len(schedule) else 'saved_paused',
                step=state['step'],planned_updates=len(schedule),engineering_only=a.engineering,evaluation=evaluation,
                frozen_before_sha256=frozen_before,frozen_after_sha256=frozen_after,trainable_groups=groups,
                restoration=restoration,wandb_url=wb.url,source_commit=commit,parent_sha256=parent['sha256'])
            atomic_json(a.output/'result.json',result)
            wb.log({'train/update':state['step'],**{'eval/'+k+'/loss':v['loss'] for k,v in evaluation.items()}})
            wb.finish()
        except Exception as exc: error=exc
    collective_error(error,device=device,phase='final receipt');dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

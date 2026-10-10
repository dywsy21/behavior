"""Replay an immutable real critic update, then test fixed-target fit capacity.

CPU-only, no simulator, actor, deployed checkpoint or new on-policy update.
Additional in-sample fits diagnose underfitting, NOT robot/generalization gains.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--update-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-steps',type=int,default=20)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.fit_steps<1:raise ValueError('Need a declared fixed-target diagnostic schedule')
    import torch
    from direct_a4_flow import ValueHead
    from critic_descent import guarded_critic_step
    torch.set_num_threads(2)
    evidence=torch.load(a.audit,map_location='cpu',weights_only=True)
    observed=json.loads(a.update_report.read_text())
    if (evidence['schema']!='critic_update_audit_v1' or evidence['actor_checkpoint']
            or evidence['training_admission']):raise ValueError('Need exact diagnostic-only preupdate evidence')
    x,y=evidence['features'],evidence['returns'];identities=evidence['identities']
    if len(identities)!=len(y) or len({r['experience_id'] for r in identities})!=len(y):
        raise ValueError('Duplicated or missing experience identity')
    def restore():
        head=ValueHead(x.shape[1]);head.load_state_dict(evidence['critic'],strict=True)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.0003)
        optimizer.load_state_dict(deepcopy(evidence['optimizer']))
        return head,optimizer
    head,optimizer=restore()
    # The actual actor+critic loss has .5 * half-MSE, not a different target.
    (.25*(head(x)-y).square().mean()).backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
    differences={}
    for name,param in head.named_parameters():
        saved=evidence['gradients'][name]
        torch.testing.assert_close(param.grad,saved,rtol=3e-4,atol=1e-6)
        differences[name]=float((param.grad-saved).abs().max())
        param.grad=saved.clone()
    lr=observed['critic_backtracking'][0]['learning_rate']
    replay=guarded_critic_step(head,optimizer,x,y,learning_rate=lr,restart_stale_momentum=True)
    if (replay['accepted_critic_lr']!=observed['accepted_critic_lr']
            or replay['critic_updated']!=observed['critic_updated']
            or len(replay['critic_backtracking'])!=len(observed['critic_backtracking'])):
        raise ValueError('CPU replay chose a different actual optimizer branch')
    for key in ('critic_half_mse_before','critic_half_mse_after'):
        if abs(replay[key]-observed[key])>1e-6:raise ValueError('Actual critic objective did not reproduce')
    def metrics(model):
        with torch.no_grad():errors=.5*(model(x)-y).square()
        groups={}
        for i,row in enumerate(identities):
            goal=row['local_subgoal'];skills=json.loads(goal['semantic_bundle'])
            key=skills[0]['verb'];groups.setdefault(key,[]).append(i)
        return dict(half_mse=float(errors.mean()),per_skill={k:dict(chunks=len(v),
            half_mse=float(errors[v].mean())) for k,v in groups.items()})
    head,optimizer=restore();curve=[dict(step=0,metrics=metrics(head))]
    for step in range(1,a.fit_steps+1):
        optimizer.zero_grad(set_to_none=True)
        (.25*(head(x)-y).square().mean()).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        report=guarded_critic_step(head,optimizer,x,y,learning_rate=lr,restart_stale_momentum=True)
        curve.append(dict(step=step,metrics=metrics(head),guard=report))
    result=dict(status='actual_gradient_and_update_reproduced',audit_sha256=file_sha(a.audit),
        actual_update_report_sha256=file_sha(a.update_report),actual_update=observed['update'],
        chunks=len(y),features=list(x.shape),gradient_max_abs_errors=differences,replayed_update=replay,
        fixed_target_fit=curve,actor_changed=False,deployable_checkpoint_written=False,
        new_simulator_steps=0,actual_rl_optimizer_updates=0,
        interpretation='Additional CPU fits reuse one frozen on-policy batch, only an in-sample critic-capacity diagnostic; not rollout learning or independent generalization.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],initial=curve[0]['metrics'],
        one_step=curve[1]['metrics'],final=curve[-1]['metrics'],actual_update_replayed=observed['update'])))


if __name__=='__main__':main()

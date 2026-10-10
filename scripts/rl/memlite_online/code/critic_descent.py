"""Backtrack a cached-observation critic step without touching the actor.

The objective and targets remain the same on-policy half-MSE. Failed trials
restore parameters AND Adam state; if none descends, publish an honest skip.
This is an optimizer safeguard, not evidence of improved robot success.
"""
from copy import deepcopy
import math
import os
from pathlib import Path

import torch


def save_critic_update_audit(path,critic,optimizer,features,returns,identities):
    """Small replayable optimizer evidence, never a deployable checkpoint."""
    if len(identities)!=len(returns) or features.requires_grad or returns.requires_grad:
        raise ValueError('One detached on-policy feature/target per bound experience required')
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    payload=dict(schema='critic_update_audit_v1',features=cpu(features),returns=cpu(returns),
        critic=cpu(critic.state_dict()),optimizer=cpu(optimizer.state_dict()),
        gradients={k:cpu(p.grad) for k,p in critic.named_parameters()},identities=identities,
        actor_checkpoint=False,training_admission=False)
    if any(v is None for v in payload['gradients'].values()):raise ValueError('Missing actual critic gradient')
    path=Path(path);tmp=path.with_suffix(path.suffix+'.pending')
    if path.exists() or tmp.exists():raise FileExistsError(path)
    with tmp.open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.link(tmp,path);tmp.unlink()


def guarded_critic_step(critic, optimizer, features, returns, *, learning_rate, attempts=7,
                       restart_stale_momentum=False):
    if (features.requires_grad or returns.requires_grad or features.ndim != 2
            or returns.shape != (features.shape[0],) or len(returns) == 0
            or not torch.isfinite(features).all() or not torch.isfinite(returns).all()
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or type(attempts) is not int or attempts < 1 or type(restart_stale_momentum) is not bool):
        raise ValueError('Pinned detached critic observations/targets and positive step required')
    parameters = list(critic.parameters())
    if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError('Need the actual finite accumulated critic gradient')
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in parameters}:
        raise ValueError('Critic optimizer contains missing or external parameters')
    with torch.no_grad():
        before = float(.5*(critic(features)-returns).square().mean())
    if not math.isfinite(before):
        raise ValueError('Nonfinite pre-update critic objective')
    saved = {k:v.detach().clone() for k,v in critic.state_dict().items()}
    optimizer_saved = deepcopy(optimizer.state_dict())
    if restart_stale_momentum and not isinstance(optimizer, (torch.optim.Adam,torch.optim.AdamW)):
        raise ValueError('Momentum restart requires an explicit Adam-family optimizer')
    trials = []; accepted = False; after = before; accepted_lr = 0.
    try:
        for attempt in range(attempts*(2 if restart_stale_momentum else 1)):
            restarted=attempt>=attempts
            if restarted:
                # If momentum points uphill on fresh on-policy targets, merely
                # halving its magnitude cannot make it a descent direction.
                # Keep variance estimates / step counts; discard ONLY stale
                # first moments for this speculative trial, not frozen actor.
                for state in optimizer.state.values():
                    if 'exp_avg' in state:state['exp_avg'].zero_()
            lr = learning_rate * .5**(attempt % attempts)
            for group in optimizer.param_groups:group['lr'] = lr
            optimizer.step()
            with torch.no_grad():
                candidate = float(.5*(critic(features)-returns).square().mean())
                finite = all(torch.isfinite(v).all() for v in critic.state_dict().values())
                changed = any(not torch.equal(v,saved[k]) for k,v in critic.state_dict().items())
                directional=sum(float((p.grad*(p-saved[name])).sum()) for name,p in critic.named_parameters())
            trial=dict(attempt=attempt,learning_rate=lr,half_mse=candidate,finite=bool(finite),changed=changed)
            if restart_stale_momentum:
                trial.update(first_moment_restarted=restarted,gradient_dot_parameter_delta=directional)
            trials.append(trial)
            if finite and math.isfinite(candidate) and candidate < before and changed:
                accepted = True;after = candidate;accepted_lr = lr;break
            critic.load_state_dict(saved,strict=True)
            optimizer.load_state_dict(deepcopy(optimizer_saved))
    except BaseException:
        critic.load_state_dict(saved,strict=True)
        optimizer.load_state_dict(optimizer_saved)
        raise
    # A failed search leaves the exact original optimizer state. A successful
    # one retains only that trial's single moment/step increment.
    if accepted:
        for group in optimizer.param_groups:group['lr'] = learning_rate
    result=dict(critic_updated=accepted,critic_half_mse_before=before,critic_half_mse_after=after,
                accepted_critic_lr=accepted_lr,critic_backtracking=trials)
    if restart_stale_momentum:
        result['critic_first_moment_restarted']=bool(accepted and trials[-1]['first_moment_restarted'])
    return result


def fit_cached_critic(critic, optimizer, features, returns, *, learning_rate, steps=1,
                     restart_stale_momentum=False):
    """Fit detached on-policy targets after ONE actor update; default unchanged.

    First step uses the actual previously accumulated/clipped gradient. Extra
    critic steps recompute only .5 * half-MSE, exactly the existing value-loss
    coefficient. They cannot re-run the actor, change GAE targets, or attach a
    gradient to cached observations. A failed descent search ends this fit.
    """
    if type(steps) is not int or steps<1:raise ValueError('Positive explicit critic fit length required')
    curve=[]
    for index in range(steps):
        if index:
            optimizer.zero_grad(set_to_none=True)
            (.25*(critic(features)-returns).square().mean()).backward()
            norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite cached critic gradient')
        trial=guarded_critic_step(critic,optimizer,features,returns,learning_rate=learning_rate,
            restart_stale_momentum=restart_stale_momentum)
        curve.append(trial)
        if not trial['critic_updated']:break
    # Preserve the entire old result/schema for default one-step recipes.
    if steps==1:return curve[0]
    accepted=[r for r in curve if r['critic_updated']]
    return dict(critic_updated=bool(accepted),critic_half_mse_before=curve[0]['critic_half_mse_before'],
        critic_half_mse_after=curve[-1]['critic_half_mse_after'],
        accepted_critic_lr=accepted[-1]['accepted_critic_lr'] if accepted else 0.,
        critic_backtracking=curve[0]['critic_backtracking'],
        critic_first_moment_restarted=any(r.get('critic_first_moment_restarted',False) for r in curve),
        critic_fit_requested_steps=steps,critic_fit_optimizer_steps=len(accepted),critic_fit_curve=curve,
        actor_steps_during_cached_fit=0,targets_recomputed_during_cached_fit=False)

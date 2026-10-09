"""Backtrack a cached-observation critic step without touching the actor.

The objective and targets remain the same on-policy half-MSE. Failed trials
restore parameters AND Adam state; if none descends, publish an honest skip.
This is an optimizer safeguard, not evidence of improved robot success.
"""
from copy import deepcopy
import math

import torch


def guarded_critic_step(critic, optimizer, features, returns, *, learning_rate, attempts=7):
    if (features.requires_grad or returns.requires_grad or features.ndim != 2
            or returns.shape != (features.shape[0],) or len(returns) == 0
            or not torch.isfinite(features).all() or not torch.isfinite(returns).all()
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or type(attempts) is not int or attempts < 1):
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
    trials = []; accepted = False; after = before; accepted_lr = 0.
    try:
        for attempt in range(attempts):
            lr = learning_rate * .5**attempt
            for group in optimizer.param_groups:group['lr'] = lr
            optimizer.step()
            with torch.no_grad():
                candidate = float(.5*(critic(features)-returns).square().mean())
                finite = all(torch.isfinite(v).all() for v in critic.state_dict().values())
                changed = any(not torch.equal(v,saved[k]) for k,v in critic.state_dict().items())
            trials.append(dict(attempt=attempt,learning_rate=lr,half_mse=candidate,finite=bool(finite),changed=changed))
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
    return dict(critic_updated=accepted,critic_half_mse_before=before,critic_half_mse_after=after,
                accepted_critic_lr=accepted_lr,critic_backtracking=trials)

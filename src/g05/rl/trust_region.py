"""Bounded Adam backtracking, preserving moments/gradients on rejected trials."""
from copy import deepcopy
import math
import torch


def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def bounded_adam_step(parameters, optimizer, evaluate, *, limit=.01, mean_limit=None,
                      scales=(1.,.1,.01,.001,.0001), on_trial=lambda row:None):
    """Try SAME clipped gradient at decreasing LR; no new environment samples.

    evaluate returns max_kl (over the entire retained rollout) and optionally
    surrogate_improved. Any exception restores weights, moments, LR and grads.
    The first accepted trial keeps its reduced LR for subsequent minibatches.
    """
    if not math.isfinite(limit) or limit <= 0 or (mean_limit is not None and
            (not math.isfinite(mean_limit) or not 0 < mean_limit <= limit)):
        raise ValueError('Positive registered KL limits required')
    parameters=list(parameters)
    if not scales or any(not 0<s<=1 for s in scales) or list(scales)!=sorted(set(scales),reverse=True):
        raise ValueError('Strictly decreasing positive step scales required')
    weights=[p.detach().cpu().clone() for p in parameters]
    gradients=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    if not any(g is not None and torch.count_nonzero(g).item() for g in gradients):
        raise ValueError('Backtracking requires a nonzero gradient')
    state=cpu_copy(optimizer.state_dict())
    rates=[group['lr'] for group in optimizer.param_groups]

    def restore():
        with torch.no_grad():
            for p,w,g in zip(parameters,weights,gradients):
                p.copy_(w)
                p.grad=None if g is None else g.clone()
        # load_state_dict may alias CPU tensors (notably Adam's CPU step
        # counter), so every restore needs a fresh deep copy of the snapshot.
        optimizer.load_state_dict(cpu_copy(state))

    trials=[]
    try:
        for scale in scales:
            restore()
            for group,rate in zip(optimizer.param_groups,rates): group['lr']=rate*scale
            optimizer.step()
            with torch.no_grad(): metrics=evaluate()
            value=float(metrics['max_kl'])
            mean=float(metrics['mean_kl']) if mean_limit is not None else None
            accepted=(math.isfinite(value) and -.002<=value<=limit
                      and (mean is None or (math.isfinite(mean) and -.002<=mean<=mean_limit))
                      and metrics.get('surrogate_improved',True))
            row=dict(scale=scale,learning_rates=[g['lr'] for g in optimizer.param_groups],
                     accepted=bool(accepted),**metrics)
            trials.append(row); on_trial(row)
            if accepted: return dict(accepted=True,scale=scale,trials=trials)
        restore()
        return dict(accepted=False,scale=None,trials=trials)
    except BaseException:
        restore()
        raise

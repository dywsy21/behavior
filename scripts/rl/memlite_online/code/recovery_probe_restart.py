"""An explicit engineering restart may not change the scientific comparison."""
from copy import deepcopy

from recovery_calibration_launch import bound,read


def validate_probe_restart(cfg, repo):
    if 'engineering_restart' not in cfg: return None
    restart=read(bound(repo,cfg['engineering_restart']))
    original=read(bound(repo,restart['original_preregistration']))
    arms={'parent':'parent','control':'control','candidate':'candidate'}
    if (restart.get('schema')!='native_learner_physical_probe_restart_v2'
            or restart['scientific_changes']!=[] or restart['arms_in_order']!=original['arms_in_order']
            or cfg.get('comparison_arm') not in arms or restart['model_node']!='lc3'
            or restart['model_gpu']!=2 or restart['simulator_gpus']!=[0,1,2]
            or restart['loopback_port']!=19820):
        raise ValueError('Unregistered scientific/resource change in restart')
    baseline=read(repo/('configs/recovery_sft/a800_native_learner_'+arms[cfg['comparison_arm']]+'_probes_v1.json'))
    observed=deepcopy(cfg);observed.pop('engineering_restart')
    # Only location and W&B grouping change. Data/models/seeds/solver/horizons,
    # original preregistration and all safety gates must remain byte-equivalent.
    observed['resource']=baseline['resource']
    observed['wandb']['group']=baseline['wandb']['group']
    if observed!=baseline: raise ValueError('Restart altered model, data, seed, horizon, solver or other science')
    return restart

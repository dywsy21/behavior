"""Decompose fixed recorded returns; never changes labels or PPO targets.

The zero-tail counterfactual is a sensitivity diagnostic, NOT a claim that
partial-episode bootstrapping is wrong for the registered continuing task.
"""
import math


def decompose_returns(records):
    if not records:raise ValueError('An actual closed episode is required')
    identity=records[0]['identity'];version=records[0]['policy_version'];sha=records[0]['policy_sha256']
    previous=records[0]['start_control'];seen=set()
    for index,row in enumerate(records):
        if (row['identity']!=identity or row['policy_version']!=version or row['policy_sha256']!=sha
                or row['experience_id'] in seen or row['start_control']!=previous
                or type(row['controls']) is not int or row['controls']<1
                or row['end_control']-row['start_control']!=row['controls']):
            raise ValueError('Cross-episode/policy, duplicated or broken physical chunk chain')
        seen.add(row['experience_id']);previous=row['end_control']
        if (any(not math.isfinite(row[k]) for k in ('reward','discount','trace_discount','old_value','next_value'))
                or not 0<=row['trace_discount']<=row['discount']<=1
                or (row['terminated'] and row['truncated'])
                or (row['terminated'] and (row['discount']!=0 or row['next_value']!=0))
                or (index<len(records)-1 and (row['terminated'] or row['truncated']))):
            raise ValueError('Invalid reward/value/boundary provenance')
        if index and abs(records[index-1]['next_value']-row['old_value'])>1e-5:
            raise ValueError('Adjacent actual state values differ')
    if not (records[-1]['terminated'] or records[-1]['truncated']):
        raise ValueError('Do not infer a boundary from an incomplete active trajectory')
    discount=1.;observed=0.
    for row in records:
        observed+=discount*row['reward'];discount*=row['discount']
    tail=discount*records[-1]['next_value']
    actual_adv=0.;zero_tail_adv=0.;chunks=[]
    for index in reversed(range(len(records))):
        row=records[index]
        delta=row['reward']+row['discount']*row['next_value']-row['old_value']
        zero_delta=delta-(row['discount']*row['next_value'] if index==len(records)-1 else 0.)
        recurse=index<len(records)-1
        actual_adv=delta+(row['trace_discount']*actual_adv if recurse else 0.)
        zero_tail_adv=zero_delta+(row['trace_discount']*zero_tail_adv if recurse else 0.)
        chunks.append(dict(experience_id=row['experience_id'],start_control=row['start_control'],
            old_value=row['old_value'],advantage=actual_adv,
            zero_tail_counterfactual_advantage=zero_tail_adv,
            final_bootstrap_advantage_component=actual_adv-zero_tail_adv))
    chunks.reverse();n=len(chunks)
    return dict(chunks=n,actual_controls=sum(r['controls'] for r in records),
        terminated=records[-1]['terminated'],truncated=records[-1]['truncated'],
        observed_discounted_reward=observed,discounted_final_bootstrap=tail,
        bootstrapped_monte_carlo_return=observed+tail,tail_state_value=records[-1]['next_value'],
        tail_absolute_share=abs(tail)/(abs(observed)+abs(tail)) if abs(observed)+abs(tail)>0 else 0.,
        mean_advantage=sum(r['advantage'] for r in chunks)/n,
        mean_final_bootstrap_advantage_component=sum(r['final_bootstrap_advantage_component'] for r in chunks)/n,
        positive_advantage_fraction=sum(r['advantage']>0 for r in chunks)/n,
        zero_tail_positive_advantage_fraction=sum(r['zero_tail_counterfactual_advantage']>0 for r in chunks)/n,
        chunk_diagnostics=chunks)

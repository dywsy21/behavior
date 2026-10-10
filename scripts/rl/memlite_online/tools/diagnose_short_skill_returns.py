"""Audit closed real reward-ledger episodes, without altering live training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from skill_return_diagnostic import decompose_returns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ledger','episodes','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    episodes=[json.loads(s) for s in a.episodes.read_text().splitlines()]
    by_id={e['job']['id']:e for e in episodes}
    if len(by_id)!=len(episodes):raise ValueError('Duplicate completed episode')
    ledger=defaultdict(list)
    for text in a.ledger.read_text().splitlines():
        row=json.loads(text);job=row['job'];transition=row['transition'];reward=row['rewards']
        if (transition['identity']['episode']!=job['id'] or transition['controls']!=len(reward)
                or reward[-1]['control_step']!=transition['end_control']):
            raise ValueError('Ledger lost job/actual-control binding')
        discount=1.;observed=0.
        for index,r in enumerate(reward):
            if (r['identity']!=transition['identity']
                    or r['control_step']!=transition['start_control']+index+1):
                raise ValueError('Broken physical reward clock')
            observed+=discount*r['reward'];discount*=r['discount']
        if abs(observed-transition['reward'])>1e-9 or abs(discount-transition['discount'])>1e-9:
            raise ValueError('Per-control rewards do not reconstruct the actual chunk target')
        ledger[job['id']].append(transition)
    results=[]
    for episode in episodes:
        records=ledger[episode['job']['id']]
        result=decompose_returns(records)
        if (result['actual_controls']!=episode['actual_controls']
                or any(r['policy_version']!=episode['policy_version'] or r['policy_sha256']!=episode['policy_sha256'] for r in records)):
            raise ValueError('Completion receipt disagrees with source trajectories')
        results.append(dict(job=episode['job'],policy_version=episode['policy_version'],
            policy_sha256=episode['policy_sha256'],success=episode['success'],outcome=episode['outcome'],**result))
    result=dict(schema='short_skill_return_decomposition_v1',ledger_sha256=file_sha(a.ledger),
        episodes_sha256=file_sha(a.episodes),status='closed_physical_rewards_and_value_terms_reconstructed',
        closed_episodes=len(results),active_or_not_yet_closed_excluded=sorted(set(ledger)-set(by_id)),episodes=results,
        live_optimizer_updates=0,changed_rewards=False,changed_timeout_labels=False,
        interpretation='Zero-tail advantages are an offline sensitivity counterfactual, not a corrected target or proof that continuing-task bootstrapping is invalid.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],closed_episodes=len(results))))


if __name__=='__main__':main()

"""Account for every preselected source, including failed/incomplete attempts.

No semantic approvals, retries, or success-rate claim. Candidate branches
from one original instance remain one independent source group.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--collections',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[];statuses=Counter();branches=Counter();groups=Counter();controls=0
    for entry in json.loads((a.sources/'manifest.json').read_text())['cases']:
        found=[root/entry['directory'] for root in a.collections if (root/entry['directory']/'source.json').exists()]
        if len(found)!=1:raise ValueError('Missing or repeated original attempt: '+entry['directory'])
        directory=found[0]
        proposal=a.sources/entry['directory']/'manifest.json'
        if (file_sha(proposal)!=entry['manifest_sha256']
                or json.loads((directory/'source.json').read_text())!=json.loads(proposal.read_text())):
            raise ValueError('Changed original source')
        path=directory/'result.json';complete=path.exists()
        if not complete:path=directory/'status.json'
        receipt=json.loads(path.read_text());status=receipt['status'] if complete else 'incomplete_quarantined'
        statuses[status]+=1;files=[];applied=0
        for stream in sorted(directory.glob('*/transitions.jsonl'))+[directory/'prefix.jsonl']:
            if not stream.exists():continue
            count=sum(1 for line in stream.read_text().splitlines() if json.loads(line))
            files.append(dict(path=str(stream),sha256=file_sha(stream),recorded_controls=count));applied+=count
        controls+=applied
        candidates=[x for x in receipt.get('branches',[]) if x['physical_recovery_candidate']]
        if candidates:groups[receipt['split']]+=1
        for branch in receipt.get('branches',[]):
            branches['physical_recovery_candidate' if branch['physical_recovery_candidate'] else branch['failure'] or branch['kind']]+=1
        rows.append(dict(case=entry['directory'],source_group=receipt['source_group'],split=receipt['split'],
            status=status,receipt_path=str(path),receipt_sha256=file_sha(path),recorded_applied_controls=applied,
            physical_recovery_branches=len(candidates),streams=files))
    result=dict(schema='recovery_collection_accounting_v1',status='all_selected_attempts_accounted',
        sources_sha256=file_sha(a.sources/'manifest.json'),source_groups=len(rows),statuses=dict(statuses),
        branch_outcomes=dict(branches),groups_with_physical_recovery_candidates=dict(groups),
        recorded_applied_controls=controls,quota=None,optimizer_steps=0,semantic_approval=False,
        scope='Local feedback teacher, not learned-actor/full-task success rate',cases=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()

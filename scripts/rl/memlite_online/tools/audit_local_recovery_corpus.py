"""Read every raw image/control/context of an offline correction corpus.

Structural validation only. This never emits semantic approvals or starts
optimization. The separate owner review and admission gate remain mandatory.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_features import feature_requests
from recovery_sft_data import CandidateArchiveReader
from validate_recovery import validate_archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    inventory=json.loads((a.corpus/'audit/inventory.json').read_text())
    summary=json.loads((a.corpus/'audit/summary.json').read_text())
    anchors=[json.loads(x) for x in (a.corpus/'audit/anchors.jsonl').read_text().splitlines()]
    history=[json.loads(x) for x in (a.corpus/'history/contexts.jsonl').read_text().splitlines()]
    receipt=json.loads((a.corpus/'history/receipt.json').read_text())
    if (summary['inventory_sha256']!=digest(inventory) or receipt['inventory_sha256']!=digest(inventory)
            or receipt['contexts_sha256']!=file_sha(a.corpus/'history/contexts.jsonl')
            or receipt['anchors_sha256']!=file_sha(a.corpus/'audit/anchors.jsonl')):
        raise ValueError('Unbound corpus/context receipt')
    counts=Counter()
    for item in inventory:
        checked=validate_archive(a.corpus/'raw'/item['path'])
        if checked['sha256']!=item['sha256']:raise ValueError('Changed archive')
        counts.update(archives=1,applied_controls=checked['controls'],decoded_images=checked['images'])
    reader=CandidateArchiveReader(a.corpus/'raw',inventory)
    by_id={h['sample_id']:h for h in history};selections=[]
    if len(by_id)!=len(anchors):raise ValueError('Incomplete / duplicated history')
    for row in anchors:
        reader.observation(row);counts['observations_read']+=1
        if row['label_audit']['full_executed_32_step_target_available']:
            reader.observation_and_actions(row);counts['clean_32_action_windows_read']+=1
        h=by_id[row['sample_id']]
        if (h['source_group']!=row['source_group'] or h['source_episode']!=row['source_episode']
                or h['control_step']!=row['control_step']):raise ValueError('Cross-instance / stale context')
        selections.append((row['sample_id'],'observable',0))
        if h['predecision'] is not None:selections.append((row['sample_id'],'predecision',0))
    requests=feature_requests(anchors,history,selections)
    planning=[r for r in requests if r['role']=='predecision']
    result=dict(schema='offline_recovery_full_read_audit_v1',status='passed',
        inventory_sha256=digest(inventory),anchors_sha256=receipt['anchors_sha256'],
        contexts_sha256=receipt['contexts_sha256'],counts=dict(counts),
        causal_requests=len(requests),planning_requests=len(planning),
        planning_check_lengths=dict(Counter(len(r['checks']) for r in planning)),
        optimizer_steps=0,semantic_approval=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

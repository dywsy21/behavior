"""CPU join of immutable candidate observations and the complete issued log."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import canonical,digest,file_sha  # noqa:E402
from recovery_history import join_causal_history  # noqa:E402
from recovery_sft_data import CandidateArchiveReader  # noqa:E402


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--logs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    inventory=json.loads((a.audit/'inventory.json').read_text())
    anchors=[json.loads(x) for x in (a.audit/'anchors.jsonl').read_text().splitlines()]
    rows,pins=[],{}
    for rank in range(8):
        path=a.logs/f'gpu_{rank}/planner_events.jsonl'
        pins[str(path.relative_to(a.logs))]=file_sha(path)
        values=[json.loads(x) for x in path.read_text().splitlines()]
        if any(v['rank']!=rank for v in values):raise ValueError('Cross-rank planner file')
        rows.extend(values)
    joined=join_causal_history(anchors,inventory,rows,CandidateArchiveReader(a.raw,inventory))
    a.output.mkdir(parents=True)
    with (a.output/'contexts.jsonl').open('x') as stream:
        for row in joined:stream.write(canonical(row)+'\n')
    receipt=dict(schema='recovery_history_join_v1',inventory_sha256=digest(inventory),
        anchors_sha256=file_sha(a.audit/'anchors.jsonl'),source_logs=pins,
        contexts_sha256=file_sha(a.output/'contexts.jsonl'),rows=len(joined),
        source_episodes=len({canonical(r['source_episode']) for r in joined}),
        served_low_context_hashes_verified=True,complete_history=True,
        labels_added=0,training_admission_unchanged=True)
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

"""Read-only archive verification after data-only transfer to A800 shared disk."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_admission import local_file  # noqa: E402
from recovery_corpus import digest,file_sha  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    started=time.monotonic()
    summary=json.loads((args.root/'audit-v4/summary.json').read_text())
    inventory=json.loads((args.root/'audit-v4/inventory.json').read_text())
    if digest(inventory)!=summary['inventory_sha256']:raise ValueError('Transferred inventory drift')
    for item in inventory:
        path=local_file(args.root/'raw-v1',item['path'])
        if path.stat().st_size!=item['bytes'] or file_sha(path)!=item['sha256']:
            raise ValueError('Bad archive transfer: '+item['path'])
    for relative,sha in summary['index_sha256'].items():
        if file_sha(local_file(args.root/'raw-v1',relative))!=sha:raise ValueError('Index changed during transfer')
    admission=json.loads((args.root/'admission-v1/admission.json').read_text())
    if admission['inventory_sha256']!=summary['inventory_sha256']:raise ValueError('Wrong admission')
    if file_sha(args.root/'audit-v4/anchors.jsonl')!=admission['anchors_sha256']:raise ValueError('Wrong candidates')
    result=dict(schema='recovery_candidate_transfer_v1',root=str(args.root),archives=len(inventory),
        bytes=sum(i['bytes'] for i in inventory),inventory_sha256=summary['inventory_sha256'],
        admission_sha256=file_sha(args.root/'admission-v1/admission.json'),seconds=time.monotonic()-started,
        verified=True,training_ready=admission['training_ready'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

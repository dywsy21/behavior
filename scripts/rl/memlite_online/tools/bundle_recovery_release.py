"""Bundle only approved-release dependencies and verify every transferred byte."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_admission import local_file
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',type=Path)
    p.add_argument('--partial-reviewed-unit',action='store_true',
        help='Transport an explicitly signed but not independently launchable unit; never changes admission gates')
    for n in ('evidence-root','corpus','admission','approvals','protected','output'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.verify:
        m=json.loads((a.verify/'transfer-manifest.json').read_text())
        for row in m['files']:
            path=local_file(a.verify,row['path'])
            if path.stat().st_size!=row['bytes'] or file_sha(path)!=row['sha256']:raise ValueError('Transferred bytes differ: '+row['path'])
        print(json.dumps(dict(status='all_transferred_files_verified',files=len(m['files']),
            bytes=sum(x['bytes'] for x in m['files']),admission_sha256=m['admission_sha256'],
            manifest_sha256=file_sha(a.verify/'transfer-manifest.json'))));return
    if not all((a.evidence_root,a.corpus,a.admission,a.approvals,a.protected,a.output)):
        p.error('Provide all build paths or --verify')
    if a.output.exists():raise FileExistsError(a.output)
    root=a.evidence_root.resolve();paths=set()
    for folder in (a.corpus,a.admission):
        paths.update(x.resolve() for x in folder.rglob('*') if x.is_file())
    paths.update([a.approvals.resolve(),a.protected.resolve()])
    admission=json.loads((a.admission/'admission.json').read_text())
    all_ready=all(v['training_ready'] for v in admission['pools'].values())
    if not all_ready and not a.partial_reviewed_unit:raise ValueError('Not a three-pool admitted release')
    if a.partial_reviewed_unit:
        from recovery_admission import audit_admission
        approved,_,verified=audit_admission(a.corpus/'audit',a.protected,a.approvals,root)
        if not any(approved.values()) or verified['pools']!=admission['pools']:
            raise ValueError('Partial unit must retain its real signed rows and unchanged admission gates')
    for line in json.loads(a.approvals.read_text())['approvals']:
        for evidence in line['evidence']:
            path=local_file(root,evidence['path'])
            if file_sha(path)!=evidence['sha256']:raise ValueError('Changed owner evidence')
            paths.add(path)
        if line['pool']=='planner':paths.add(local_file(root,line['label']['verified_plan_path']))
    files=[dict(path=str(x.relative_to(root)),bytes=x.stat().st_size,sha256=file_sha(x)) for x in sorted(paths)]
    manifest=dict(schema='recovery_release_transfer_v1',files=files,admission_sha256=file_sha(a.admission/'admission.json'),
                  source_evidence_root=str(root),contains_weights=False,contains_credentials=False,
                  partial_reviewed_unit=a.partial_reviewed_unit,all_three_pools_ready=all_ready,
                  grants_training_launch_permission=False)
    encoded=(json.dumps(manifest,indent=2)+'\n').encode()
    with tarfile.open(a.output,'x:gz') as archive:
        info=tarfile.TarInfo('transfer-manifest.json');info.size=len(encoded);archive.addfile(info,io.BytesIO(encoded))
        for row in files:archive.add(root/row['path'],arcname=row['path'],recursive=False)
    print(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in files),archive_bytes=a.output.stat().st_size,
        archive_sha256=file_sha(a.output),admission_sha256=manifest['admission_sha256'])))


if __name__=='__main__':main()

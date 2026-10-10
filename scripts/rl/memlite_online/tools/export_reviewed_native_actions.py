"""Export only explicitly owner-reviewed native learner action windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import file_sha
from recovery_native_actions import export_native_bundle,NativeLearnerActionReader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','owner-review','review-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Frozen clean export source required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    manifest=export_native_bundle(a.config,a.owner_review,a.review_root,a.output,source_commit=commit)
    sha=file_sha(a.output/'manifest.json')
    reader=NativeLearnerActionReader(a.output,sha,file_sha(a.owner_review))
    for i in range(len(reader.rows)):reader.read(i)
    print(json.dumps(dict(status='all_native_samples_roundtripped_no_training',samples=len(reader.rows),
        source_groups=len(manifest['source_groups']),manifest_sha256=sha,bytes=sum(f['bytes'] for f in manifest['files']))))


if __name__=='__main__':main()

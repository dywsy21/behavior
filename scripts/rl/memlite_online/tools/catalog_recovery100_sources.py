"""Stream all accepted original metadata and inventory every task/skill/split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_coverage import build_catalog


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','protected','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    recipe=json.loads(a.recipe.read_text());release=Path(recipe['root'])/recipe['expert_release']
    acceptance=json.loads((release/'acceptance.json').read_text())
    if (acceptance['status']!='ACCEPTED' or not all(acceptance['gates'].values())
            or file_sha(release/'manifest.json')!=acceptance['manifest_sha256']):raise ValueError('Unaccepted source release')
    protected=set(json.loads(a.protected.read_text())['groups'])
    with (release/'episodes.jsonl').open() as stream:catalog=build_catalog((json.loads(x) for x in stream),protected)
    if len(catalog['tasks'])!=100:raise ValueError('Original source is not the 100-task release')
    a.output.mkdir(parents=True)
    queue=catalog.pop('queue')
    with (a.output/'queue.jsonl').open('x') as stream:
        for row in queue:stream.write(json.dumps(row,sort_keys=True)+'\n')
    catalog.update(original_release_sha256=acceptance['manifest_sha256'],protected_sha256=file_sha(a.protected),
        queue_sha256=file_sha(a.output/'queue.jsonl'),queue_rows=len(queue))
    (a.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(dict(tasks=len(catalog['tasks']),source_groups=catalog['source_groups'],queue_rows=len(queue),
        verbs=sorted({r['verb'] for r in queue}),grasp_tasks=sum('GRASP' in t['verbs'] for t in catalog['tasks']),
        catalog_sha256=file_sha(a.output/'catalog.json'),output=str(a.output))))


if __name__=='__main__':main()

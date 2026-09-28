"""Register a no-sampling, no-accepted-update numerical diagnostic."""
import json
import math
from pathlib import Path
from common import PREVIOUS,OUT,commit,save


def main():
    manifest=json.loads((PREVIOUS/'manifest.json').read_text())
    supervisor=json.loads((PREVIOUS/'supervisor.json').read_text())
    status=json.loads((PREVIOUS/'status.json').read_text())
    for pid in (supervisor['supervisor'],supervisor['learner']):
        path=Path(f'/proc/{pid}/stat')
        if path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise ValueError('Previous update still active')
    if status['actor_updates'] or status['controls'] or not status['trust_region_exhausted']:
        raise ValueError('Expected fully rolled-back update')
    elapsed=manifest['prior_active_seconds']+supervisor['seconds']
    if 7200-math.ceil(elapsed)<180: raise ValueError('Insufficient original time budget')
    manifest.update(source_commit=commit(),continued_from=str(PREVIOUS),prior_active_seconds=elapsed,
        max_active_wall_seconds=min(600,7200-math.ceil(elapsed)),entry='precision_probe',
        probe=dict(indices=[0,3,6,7,20,38],learning_rates=[0.,1e-10,1e-8,1e-7],
                   accepted_optimizer_steps=0,extra_environment_controls=0))
    OUT.mkdir(parents=True,exist_ok=False); save(OUT/'manifest.json',manifest)
    print(json.dumps(dict(output=str(OUT),prior_controls=manifest['prior_controls'],prior_seconds=elapsed,
                         wall_limit=manifest['max_active_wall_seconds'],extra_controls=0)))


if __name__=='__main__': main()

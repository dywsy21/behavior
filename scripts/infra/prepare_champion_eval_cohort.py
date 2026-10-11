"""Freeze original-heldout eval identities only; no RGB/model/inference work."""
import argparse
from collections import Counter
import json
from pathlib import Path

from behavior_champion.dataset import NativeV3Dataset
from behavior_champion.evaluation import FixedEvalPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = NativeV3Dataset(args.release, 'eval')
    layout = json.loads(args.layout.read_text())['layout']
    plan = FixedEvalPlan(data, layout['dataset_to_model'])
    if (len(data.episodes) != 994 or data.excluded_episode_counts != {'train': 105, 'eval': 6} or
            set(Counter(row['task'] for row in plan.identities).values()) != {16}):
        raise ValueError('Original protected eval scope changed')
    result = dict(status='fixed_original_heldout_cohort_selected_not_evaluated',
        contract=plan.contract, plan_sha256=plan.fingerprint, identities=plan.identities,
        optimizer_updates=0, model_forwards=0, physical_controls=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'identities'}))


if __name__ == '__main__': main()

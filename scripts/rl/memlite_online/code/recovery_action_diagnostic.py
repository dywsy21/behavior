"""Event-balanced, matched-noise FM diagnostic aggregation (not success)."""
from collections import defaultdict
import math


def summarize_action_losses(rows, expected_sample_ids, seeds):
    expected = {(sample, seed) for sample in expected_sample_ids for seed in seeds}
    keys = [(r['sample_id'], r['noise_seed']) for r in rows]
    if not expected or len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Missing, duplicate or foreign sample/noise measurement')
    samples = defaultdict(list)
    for row in rows:
        if (row['split'] not in ('train', 'dev') or not row['mechanism']
                or not all(math.isfinite(row[k]) for k in ('numerator', 'denominator'))
                or row['numerator'] < 0 or row['denominator'] <= 0):
            raise ValueError('Invalid admitted objective row')
        samples[row['sample_id']].append(row)
    scopes = defaultdict(lambda: defaultdict(list))
    for sample_rows in samples.values():
        first = sample_rows[0]
        if any(any(row[k] != first[k] for k in ('source_group', 'event_id', 'split', 'mechanism', 'control_step'))
               for row in sample_rows):
            raise ValueError('Mixed identity within one sample')
        loss = sum(r['numerator'] for r in sample_rows)/sum(r['denominator'] for r in sample_rows)
        scopes[(first['split'], first['mechanism'])][(first['source_group'], first['event_id'])].append(loss)
    return [dict(split=split, mechanism=mechanism, events=len(events), anchors=sum(map(len, events.values())),
                 event_balanced_loss=sum(sum(v)/len(v) for v in events.values())/len(events),
                 anchor_mean_loss=sum(sum(v) for v in events.values())/sum(map(len, events.values())))
            for (split, mechanism), events in sorted(scopes.items())]

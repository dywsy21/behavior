"""Partition independent planner calls; never batch or share episode memory."""
import copy

from recovery_corpus import digest


def generation_shard(schedule, shard_index=0, num_shards=1):
    if (type(shard_index) is not int or type(num_shards) is not int
            or not 1 <= num_shards <= len(schedule)
            or not 0 <= shard_index < num_shards):
        raise ValueError('Invalid nonempty generation shard')
    if len({tuple(row) for row in schedule}) != len(schedule):
        raise ValueError('Duplicated generation schedule entry')
    selected = [(i, *schedule[i]) for i in range(shard_index, len(schedule), num_shards)]
    binding = dict(protocol='independent_planner_generation_shards_v1',
                   shard_index=shard_index, num_shards=num_shards,
                   global_schedule=[list(row) for row in schedule],
                   selected_schedule_sha256=digest(selected))
    return selected, binding


def merge_generation_shards(parts):
    """Reject missing, duplicate, foreign, or partial shards before aggregating.

    parts are (manifest, terminal result, actual rows). No inferred success from
    process exit, no reweighting unequal shard averages, no cross-model mixing.
    """
    if not parts:
        raise ValueError('No generation shards')
    first = copy.deepcopy(parts[0][0])
    binding = first.pop('sharding', None)
    if not binding or binding.get('protocol') != 'independent_planner_generation_shards_v1':
        raise ValueError('Unregistered shard protocol')
    schedule = binding['global_schedule']
    count = binding['num_shards']
    if digest(schedule) != first['schedule_sha256']:
        raise ValueError('Global schedule hash mismatch')
    if len(parts) != count:
        raise ValueError('Missing or extra generation shards')
    seen = set()
    rows = {}
    seconds = []
    for manifest, result, actual in parts:
        common = copy.deepcopy(manifest)
        shard = common.pop('sharding', None)
        if common != first or not shard or shard.get('num_shards') != count:
            raise ValueError('Foreign model, source, config, or schedule')
        index = shard.get('shard_index')
        selected, expected = generation_shard(schedule, index, count)
        if shard != expected or index in seen:
            raise ValueError('Duplicate or changed generation shard')
        seen.add(index)
        if (result.get('status') != 'completed'
                or any(result.get(k) != v for k, v in manifest.items())):
            raise ValueError('Incomplete or mismatched terminal receipt')
        if ('model_before_sha256' in manifest
                and result.get('model_after_sha256') != manifest['model_before_sha256']):
            raise ValueError('Read-only generation model changed or final digest is missing')
        if len(actual) != len(selected):
            raise ValueError('Missing or extra generated rows')
        for row, (position, split, sample_index) in zip(actual, selected):
            if (row.get('global_schedule_index') != position
                    or row.get('scheduled_sample_index') != sample_index
                    or row.get('split') != split or position in rows):
                raise ValueError('Reordered, duplicated, or misbound generated row')
            rows[position] = row
        seconds.append(result['seconds'])
    if sorted(rows) != list(range(len(schedule))):
        raise ValueError('Incomplete global generation coverage')
    first['sharding'] = dict(protocol=binding['protocol'], num_shards=count,
                             merged=True, global_schedule=schedule)
    if 'model_before_sha256' in first:
        first['model_after_sha256'] = first['model_before_sha256']
    return first, [rows[i] for i in range(len(schedule))], seconds

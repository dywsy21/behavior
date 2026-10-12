"""Restricted, reviewed period128 planner augmentation; never low BC/results.

Reuses original candidate IDs, frames and preprocessing. Only causal command
history is replaced. The original recovery event/rank order can be replayed
exactly while replacing expert anchors, isolating the registered augmentation.
"""
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random

from recovery_corpus import file_sha
from recovery_periodic_teacher import validate_candidate_projection

REPO = Path(__file__).resolve().parents[4]
PROTOCOL = 'reviewed_annotation_period128_v1'
ADMISSION_STATUS = 'ACCEPTED_RESTRICTED_HIGH_INTENT_AUGMENTATION_NOT_A_TRAINING_LAUNCH'


def bound(base, relative):
    base = Path(base).resolve(); path = (base / relative).resolve()
    if not path.is_relative_to(base): raise ValueError('Foreign periodic artifact')
    return path


def validate_periodic_row(row):
    identity, projection = row['identity'], row['projection']
    if (identity['split'] != 'train' or row['human_admission'] is not False
            or row['teacher_kind'] != 'annotation_periodic_command_trace_not_actor_rollout'
            or row['annotation_trace']['period'] != 128
            or projection['memlite_branch'] != 'high' or projection['schema_version'] != 6
            or projection['outcome_supervision_mask'] is not False
            or projection['low_action_supervision_mask'] is not False
            or projection['known_previous_outcome'] != 'UNKNOWN'
            or projection['outcome_target'] != 'UNKNOWN' or projection['task_complete'] is not False
            or projection['next_decision'] != 'EXECUTE'):
        raise ValueError('Periodic augmentation cannot create test, low-action or outcome labels')
    return int(identity['original_candidate_index']), int(identity['task'])


class PeriodicTeacherPool:
    def __init__(self, root, spec, *, repo=REPO):
        if spec.get('protocol') != PROTOCOL: raise ValueError('Unknown periodic teacher protocol')
        self.spec = spec
        self.admission_path = bound(repo, spec['admission'])
        self.candidate_path = bound(root, spec['candidates'])
        self.control_path = bound(root, spec['control_schedule']['path'])
        for path, expected in ((self.admission_path, spec['admission_sha256']),
                               (self.candidate_path, spec['candidate_sha256']),
                               (self.control_path, spec['control_schedule']['sha256'])):
            if file_sha(path) != expected: raise ValueError('Changed periodic teacher input: ' + str(path))
        admission = json.loads(self.admission_path.read_text())
        scope = admission['admission_scope']
        if (admission.get('status') != ADMISSION_STATUS or admission.get('training_admitted') is not True
                or admission['candidate']['sha256'] != spec['candidate_sha256']
                or scope['allowed_component'] != 'high_planner_only original-TRAIN intent augmentation'
                or scope['required_outcome_supervision_mask'] is not False
                or scope['required_low_action_supervision_mask'] is not False
                or admission['human_review']['personally_reviewed_cases'] != 120
                or admission['human_review']['tasks'] != 100):
            raise ValueError('Missing restricted personal-review admission')
        self.admission = admission
        from behavior_champion.release import AcceptedRelease
        self.release = AcceptedRelease(bound(root, admission['candidate']['source_release']),
            manifest_sha256=admission['candidate']['source_manifest_sha256'])
        for name in ('train_candidates.npy', 'train_tasks.npy', 'episode_offsets.npy'):
            self.release.verify_file(name)
        for key in ('real_cpu_processor', 'real_cpu_token_and_loss_mask'):
            evidence = admission[key]
            result_path = bound(root, str(Path(evidence['run']) / 'result.json'))
            if file_sha(result_path) != evidence['result_sha256']:
                raise ValueError('Changed admitted real processor/token evidence')
            result = json.loads(result_path.read_text())
            if (result.get('status') != 'passed_cpu_causal_source_and_processor_contract'
                    or result['candidate_sha256'] != spec['candidate_sha256']
                    or result['reconstructed_candidates'] != 1800 or result['processed_review_cases'] != 120):
                raise ValueError('Incomplete original processor acceptance')
        self.rows = [json.loads(line) for line in self.candidate_path.read_text().splitlines()]
        self.by_index, self.by_task = {}, defaultdict(list)
        kinds = Counter()
        for row in self.rows:
            index, task = validate_periodic_row(row)
            if index in self.by_index: raise ValueError('Duplicate periodic source anchor')
            self.by_index[index] = row; self.by_task[str(task)].append(index)
            kinds[task, row['identity']['kind']] += 1
        expected = Counter({(t, k): n for t in range(100)
                            for k,n in (('initial',2),('transition',8),('continuation',8))})
        if len(self.rows) != 1800 or kinds != expected:
            raise ValueError('Periodic selection missing tasks, strata or source anchors')
        self.handoff = None
        if spec.get('learned_handoff') is not None:
            from recovery_handoff_data import HandoffFeedbackPool
            self.handoff = HandoffFeedbackPool(root, spec['learned_handoff'], self, repo=repo)


def with_periodic_teacher(dataset_class, pool):
    class ReviewedPeriodicPlanner(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.branch != 'high' or self.split != 'train' or self.release.resolve() != pool.release.root.resolve():
                raise ValueError('Periodic TRAIN augmentation is high-only on the accepted source')
            self.periodic_by_task = dict(pool.by_task)
            self._periodic_projection_cache = {}

        def raw(self, index, *, images=True):
            import torch
            index = int(index)
            if index not in pool.by_index: raise ValueError('Unreviewed periodic source index')
            row = pool.by_index[index]
            if index not in self._periodic_projection_cache:
                pair = self.candidates[index]
                value = validate_candidate_projection(row, self.episode(int(pair[0])),
                    original_index=index, original_pair=pair)
                if int(self.task_ids[index]) != row['identity']['task']:
                    raise ValueError('Periodic source task differs from original index')
                if getattr(pool, 'handoff', None) is not None:
                    value = pool.handoff.projection(index, value)
                self._periodic_projection_cache[index] = value
            raw, identity = super().raw(index, images=images)
            if any(identity[key] != row['identity'][key] for key in ('episode','task','frame','raw_episode','instance')):
                raise ValueError('Periodic label and actual decoded observation differ')
            raw.update(model_projection=copy.deepcopy(self._periodic_projection_cache[index]),
                action_is_pad=torch.ones(32, dtype=torch.bool),
                action={m['key']:torch.zeros(32, m['raw_shape']) for m in self.config['raw_shape']['action']})
            identity['planner_history_protocol'] = PROTOCOL
            return raw, identity
    return ReviewedPeriodicPlanner


def remap_control_schedule(control, pool_by_task, task_for_index, *, seed, recovery_rows, binding):
    """Keep exact recovery examples, rank slots, events and pass order.

    Independently shuffled per-task periodic pools avoid random-number coupling
    between replacement expert anchors and the unchanged recovery schedule.
    """
    for key in ('component','parent_sha256','admission_sha256','world_size','micro_batch'):
        if control['binding'][key] != binding[key]: raise ValueError('Different paired control binding: ' + key)
    pools = {str(k): list(v) for k,v in pool_by_task.items()}
    if not pools or any(not values or len(values) != len(set(values)) for values in pools.values()):
        raise ValueError('Empty or repeated periodic source pool')
    for task, values in sorted(pools.items()): random.Random(f'{seed}:periodic:{task}').shuffle(values)
    counters = Counter(); schedule = []
    for batch in control['schedule']:
        mapped = copy.deepcopy(batch); rows = []; tasks = []; observed_events = []
        for kind, index in batch['rows']:
            if kind == 'expert':
                task = str(int(task_for_index(index))); tasks.append(task)
                if task not in pools: raise ValueError('Control task missing from periodic pool')
                chosen = pools[task][counters[task] % len(pools[task])]; counters[task] += 1
                rows.append(['expert', chosen])
            elif kind == 'new':
                if not 0 <= index < len(recovery_rows): raise ValueError('Different recovery row table')
                item = recovery_rows[index]
                if item['candidate']['split'] != 'train': raise ValueError('Recovery holdout leak')
                observed_events.append([item['candidate']['source_group'], item['approval']['event_id']])
                rows.append(['new', index])
            else: raise ValueError('Unknown source in paired schedule')
        if (len(rows) != 8 or len(tasks) != 6 or len(set(tasks)) != 6
                or sorted(observed_events) != sorted(batch['new_events'])
                or batch['expert_count'] != 6 or batch['new_count'] != 2):
            raise ValueError('Expected original six-task/two-recovery paired batch')
        mapped['rows'] = rows; schedule.append(mapped)
    return schedule


def validate_periodic_launch(ticket, recipe):
    if ticket['component'] != 'H1' or ticket.get('engineering_smoke'):
        raise ValueError('Periodic augmentation requires a separate real high-planner ticket')
    if (ticket['world_size'],ticket['micro_batch'],ticket['global_batch'],ticket['maximum_updates'],ticket['event_passes'])!=(8,1,8,360,20):
        raise ValueError('Keep the registered same-parent paired360 exposure and eight-rank batch')
    spec = recipe['H1']['periodic_teacher']; files = ticket['files']
    expected = {'periodic_admission':spec['admission_sha256'], 'periodic_candidates':spec['candidate_sha256'],
                'periodic_control_schedule':spec['control_schedule']['sha256']}
    learned = spec.get('learned_handoff') is not None
    audit_key = 'handoff_processor_audit' if learned else 'periodic_processor_audit'
    if not (set(expected) | {audit_key}) <= files.keys():
        raise ValueError('Missing periodic real-adapter audit or source binding')
    for key, digest in expected.items():
        if files[key]['sha256'] != digest or file_sha(files[key]['path']) != digest:
            raise ValueError('Different periodic pinned input: ' + key)
    if recipe['H1'].get('original_feedback') != 'causal_expert_unknown_v1':
        raise ValueError('Keep the unchanged original causal-feedback evaluation control')
    if learned:
        from recovery_handoff_data import validate_handoff_launch
        validate_handoff_launch(ticket, recipe)
        return
    audit = json.loads(Path(files['periodic_processor_audit']['path']).read_text())
    if (audit.get('schema') != 'periodic_planner_processor_audit_v1' or audit.get('real_adapter_used') is not True
            or audit.get('source_commit') != ticket['source_commit']
            or audit.get('recipe_sha256') != files['recipe']['sha256']
            or audit.get('periodic_admission_sha256') != spec['admission_sha256']
            or audit.get('candidate_sha256') != spec['candidate_sha256']
            or audit.get('status') != 'passed_cpu_causal_source_and_processor_contract'
            or audit.get('reconstructed_candidates') != 1800 or audit.get('processed_review_cases') != 120
            or audit.get('tokenized_review_cases') != 120 or audit.get('optimizer_updates') != 0):
        raise ValueError('Periodic training lacks the actual source-bound adapter/token audit')
    if recipe['H1'].get('original_feedback') != 'causal_expert_unknown_v1':
        raise ValueError('Keep the unchanged original causal-feedback evaluation control')


def checkpoint_reservation(parent_bytes, updates, checkpoint_every, champion_reserve):
    if min(parent_bytes, updates, checkpoint_every) <= 0 or champion_reserve < (775 << 30):
        raise ValueError('Invalid periodic checkpoint or champion space reservation')
    per_file = math.ceil(parent_bytes * 1.10)
    planned = math.ceil(updates / checkpoint_every) + 1  # plus arbitrary-step emergency
    return dict(planned_checkpoints=planned, checkpoint_upper_bytes=per_file,
        required_free_bytes=champion_reserve + planned * per_file,
        runtime_stop_free_bytes=champion_reserve + 2 * per_file)

"""Verified rollout action adapter and finite event-balanced SFT schedule.

Archives remain immutable. Only an externally pinned, ready ACTION pool may
instantiate the training dataset. Labels/audits/physics never enter samples.
"""
from collections import defaultdict, OrderedDict
import io
import json
from pathlib import Path
import random
import zipfile

from recovery_admission import local_file
from recovery_corpus import canonical, file_sha


def require_training_pool(release, pool, expected_admission_sha256, *, purpose='training'):
    release = Path(release)
    if file_sha(release / 'admission.json') != expected_admission_sha256:
        raise ValueError('Stale or unpinned training admission')
    receipt = json.loads((release / 'admission.json').read_text())
    if receipt.get('schema') != 'recovery_admission_v1' or pool not in receipt['pools']:
        raise ValueError('Unknown training pool')
    gate = receipt['pools'][pool]
    if gate.get('training_ready') is not True or gate['blockers']:
        raise ValueError('Recovery training BLOCKED: ' + pool + ': ' + '; '.join(gate['blockers']))
    for name, sha in receipt['files'].items():
        if file_sha(local_file(release, name)) != sha:
            raise ValueError('Approved rows changed after admission')
    rows = [json.loads(line) for line in (release / (pool + '.jsonl')).read_text().splitlines()]
    if not rows or any(r['approval']['pool'] != pool for r in rows):
        raise ValueError('Empty or wrong objective pool')
    from recovery_evaluation_partition import rows_for_purpose
    partition = None
    name = receipt.get('evaluation_partition_file')
    if name is not None:
        if name not in receipt['files']:
            raise ValueError('Unbound evaluation partition')
        partition = json.loads(local_file(release, name).read_text())
    selected = rows_for_purpose(rows, partition, purpose)
    if partition is not None and purpose == 'training':
        # A held-out test cohort may not supply the minimum DEV count/classes
        # needed to admit a training experiment.
        for split, minimum in (('train', 8), ('dev', 4)):
            local = [r for r in selected if r['candidate']['split'] == split]
            if (len({r['approval']['event_id'] for r in local}) < minimum
                    or len({r['candidate']['source_group'] for r in local}) < 2):
                raise ValueError('Frozen-test rows cannot satisfy training/selection coverage')
            if pool == 'outcome' and not {'IN_PROGRESS','SUCCEEDED','FAILED'} <= {
                    r['approval']['label']['value'] for r in local}:
                raise ValueError('Frozen-test rows cannot supply missing training/selection classes')
    return receipt, selected


class CandidateArchiveReader:
    """CPU diagnostic/source reader, not an approval or a training Dataset."""
    def __init__(self, root, inventory):
        self.root = Path(root)
        self.inventory = inventory
        self.by_episode = defaultdict(list)
        for item in inventory:
            self.by_episode[canonical([item['episode']['run'],item['episode']['episode_id']])].append(item)
        self.cache = OrderedDict()

    def episode(self, identity):
        key = canonical(identity)
        if key not in self.cache:
            merged, headers = {}, {}
            for item in self.by_episode[key]:
                path = local_file(self.root, item['path'])
                if file_sha(path) != item['sha256']:
                    raise ValueError('Source archive changed after inventory')
                with zipfile.ZipFile(path) as archive:
                    header = json.loads(archive.read('manifest.json'))
                    if [header['episode']['run'],header['episode']['episode_id']] != identity:
                        raise ValueError('Cross-episode source archive')
                    headers[item['path']] = header
                    for line in archive.read('transitions.jsonl').splitlines():
                        row = json.loads(line)
                        step = row['control_step']
                        if step in merged and canonical(merged[step]) != canonical(row):
                            raise ValueError('Conflicting overlap')
                        merged[step] = row
            if not merged:
                raise ValueError('No archived evidence for source episode')
            self.cache[key] = (merged, headers)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def observation(self, candidate):
        """Read s[t] without requiring future action availability (H0/dev tails)."""
        import numpy as np
        from PIL import Image
        rows, headers = self.episode(candidate['source_episode'])
        t = candidate['control_step']
        reference = candidate['actor_input']['rgb']
        header = headers[reference['archive']]
        if reference['control_step'] != t or header['rgb_anchors'][str(t)]['sha256'] != reference['sha256']:
            raise ValueError('Observation/source image mismatch')
        observation_only = candidate['actor_input'].get('observation_only', False)
        if observation_only:
            final = header.get('terminal_observation')
            if (t in rows or final is None or final['control_step'] != t
                    or final['has_next_executed_action'] is not False
                    or candidate['label_audit']['full_executed_32_step_target_available']):
                raise ValueError('Final observation confused with an applied action')
            proprio = final['proprio']
        else:
            proprio = rows[t]['proprio_before']
        state = np.asarray(proprio, dtype=np.float32)[None]
        if proprio != candidate['actor_input']['proprio_before']:
            raise ValueError('Pre-action proprio was modified')
        if state.shape != (1,61) or not np.isfinite(state).all():
            raise ValueError('Malformed real robot proprio tensors')
        images = {}
        with zipfile.ZipFile(local_file(self.root, reference['archive'])) as archive:
            for camera in ('head_rgb','left_wrist_rgb','right_wrist_rgb'):
                data = archive.read(f'rgb/{t:08d}/{camera}.jpg')
                import hashlib
                if hashlib.sha256(data).hexdigest() != reference['sha256'][camera]:
                    raise ValueError('RGB content mismatch')
                im = np.asarray(Image.open(io.BytesIO(data)).convert('RGB')).copy()
                if im.shape != (224,224,3):
                    raise ValueError('Unexpected original RGB dimensions')
                images[camera] = im
        return state, images

    def observation_and_actions(self, candidate):
        import numpy as np
        state, images = self.observation(candidate)
        rows, _ = self.episode(candidate['source_episode'])
        t = candidate['control_step']
        if candidate['actor_input'].get('observation_only',False):
            raise ValueError('Final observation has no next action; cannot be BC')
        if not candidate['label_audit']['full_executed_32_step_target_available']:
            raise ValueError('Incomplete contiguous same-intent action target')
        expected = candidate['actor_input']['issued_skills_semantic_json']
        future = [rows[step] for step in range(t,t+32)]
        if (any(canonical(json.loads(r['context']['active_skills_semantic_json'])) != expected or
                r['context']['parent_goal'] != candidate['actor_input']['parent_goal'] for r in future)
                or any(r['terminated'] or r['truncated'] for r in future[:-1])):
            raise ValueError('Action chunk crossed a skill switch or terminal')
        if any(r.get('simulator_apply_ack') is not True for r in future):
            raise ValueError('Unacknowledged simulator action cannot become a BC target')
        if any(r.get('label_kind')=='injected_fault_not_BC' for r in future):
            raise ValueError('Injected perturbation is never a positive action target')
        action = np.asarray([r['action_executed_raw23'] for r in future], dtype=np.float32)
        if action.shape != (32,23) or not np.isfinite(action).all():
            raise ValueError('Malformed real robot action tensors')
        return state, action, images


def raw_observation(candidate, reader, config):
    """Whitelisted single-frame RGB/proprio; never copy labels or physics."""
    import torch
    state, images = reader.observation(candidate)
    return dict(task=candidate['task'].replace('_',' '),idx=0,embodiment='galaxea_r1pro',frequency=30,
        state_is_pad=torch.zeros(1,dtype=torch.bool),image_is_pad=torch.zeros(1,dtype=torch.bool),
        images={k:torch.from_numpy(v).permute(2,0,1)[None] for k,v in images.items()},
        state={m['key']:torch.from_numpy(state[:,m['start_index']:m['start_index']+m['raw_shape']].copy())
               for m in config['raw_shape']['state']})


class VerifiedRecoveryActionDataset:
    def __init__(self, release, raw_root, inventory, config, *, split, admission_sha256):
        from recovery_corpus import digest
        self.receipt, rows = require_training_pool(release, 'action', admission_sha256)
        if digest(inventory) != self.receipt['inventory_sha256']:
            raise ValueError('Training data does not match pinned archive inventory')
        if split not in ('train','dev'):
            raise ValueError('No public/protected pool is a training source')
        self.rows = [r for r in rows if r['candidate']['split'] == split]
        if not self.rows:
            raise ValueError('No verified rows for this split')
        self.reader = CandidateArchiveReader(raw_root, inventory)
        self.config, self.split, self.processor = config, split, None

    def __len__(self):
        return len(self.rows)

    def raw(self, index):
        import torch
        from g05.utils.memlite_skill_protocol import parse_active_skills_semantic_json, semantic_active_skills_text
        from g05.data.memlite_stage1_labels import projection
        item = self.rows[index]
        row, approval = item['candidate'], item['approval']
        if (approval['label'] != dict(quality='verified_correct_execution',executed_controls=32)
                or approval['sample_id'] != row['sample_id'] or approval['reviewed_end'] < row['control_step']+32
                or 'binding_quarantine' in row['label_audit']):
            raise ValueError('Invalid approved ACTION row')
        state, action, images = self.reader.observation_and_actions(row)
        semantic = row['actor_input']['issued_skills_semantic_json']
        task = row['task'].replace('_',' ')
        segment = dict(parent=row['actor_input']['parent_goal'],semantic=semantic,parent_supervised=False,
                       text=semantic_active_skills_text(parse_active_skills_semantic_json(semantic)))
        # Low prefix uses only task/parent/current bundle, images and proprio.
        # Empty history is not a synthetic high-planner training sample.
        raw = dict(task=task,idx=index,embodiment='galaxea_r1pro',frequency=30,
            action_is_pad=torch.zeros(32,dtype=torch.bool),state_is_pad=torch.zeros(1,dtype=torch.bool),
            image_is_pad=torch.zeros(1,dtype=torch.bool),
            images={k:torch.from_numpy(v).permute(2,0,1)[None] for k,v in images.items()},
            model_projection=projection(segment,branch='low',task_name=task,previous_intent='None',previous_parent='none',history=[]))
        for domain,values in (('action',action),('state',state)):
            raw[domain] = {m['key']:torch.from_numpy(values[:,m['start_index']:m['start_index']+m['raw_shape']].copy())
                           for m in self.config['raw_shape'][domain]}
        return raw

    def __getitem__(self,index):
        import numpy as np
        import torch
        from g05.utils.training.stage1_model import make_processor
        if self.processor is None:
            self.processor = make_processor(self.config, training=self.split=='train')
        row = self.rows[index]['candidate']
        old_random,old_np = random.getstate(),np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                seed = int(row['sample_id'][:8],16)
                torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                sample = self.processor.preprocess(self.raw(index))
        finally:
            random.setstate(old_random);np.random.set_state(old_np)
        if tuple(torch.nonzero(sample['action_dim_is_pad']).flatten().tolist()) != (7,8,17,18):
            raise ValueError('Lost base/trunk or changed R1Pro padding')
        if sample['action'].shape != (32,27) or not torch.isfinite(sample['action']).all():
            raise ValueError('Invalid inherited normalized action')
        sample['source_identity'] = dict(pool='verified_action',sample=row['sample_id'],
            event=self.rows[index]['approval']['event_id'],task=row['task'],source_group=row['source_group'])
        return sample


def finite_mixture_schedule(new_rows, expert_by_task, *, batch_size=64, maximum_event_passes=5, seed=17, pool='action'):
    """At most ONE anchor/event/pass, ~70/30 expert/new, no repeat-filled tail.

    The schedule is global and committed before DDP splits it. Reuse is counted
    by physical event, not by highly correlated frame count; small corpora yield
    few updates rather than being duplicated up to a requested step budget.
    """
    if batch_size < 8 or not 1 <= maximum_event_passes <= 5 or pool not in ('action','planner'):
        raise ValueError('Invalid finite pilot budget')
    new_by_event = defaultdict(list)
    for index,item in enumerate(new_rows):
        if item['candidate']['split'] != 'train' or item['approval']['pool'] != pool:
            raise ValueError('Only accepted train rows of the requested pool may enter mixture')
        new_by_event[(item['candidate']['source_group'],item['approval']['event_id'])].append(index)
    if not new_by_event or len(expert_by_task) < 2 or any(not v for v in expert_by_task.values()):
        raise ValueError('Need approved events and at least two expert tasks')
    rng = random.Random(seed)
    new_per_batch = max(1, round(batch_size*.3))
    tasks = sorted(expert_by_task)
    for epoch in range(maximum_event_passes):
        events = list(new_by_event)
        rng.shuffle(events)
        for start in range(0,len(events),new_per_batch):
            selected = events[start:start+new_per_batch]
            new = [('new',rng.choice(new_by_event[e])) for e in selected]
            expert_count = batch_size-len(new) if len(new) == new_per_batch else round(len(new)*7/3)
            rng.shuffle(tasks)
            experts = [('expert',rng.choice(expert_by_task[tasks[i%len(tasks)]])) for i in range(expert_count)]
            batch = new+experts
            rng.shuffle(batch)
            yield dict(event_pass=epoch, rows=batch,new_events=[list(e) for e in selected],
                       new_count=len(new),expert_count=len(experts))

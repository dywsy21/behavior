"""Execution-only bookkeeping. No reward, success, object state or model input.

The official evaluator calls ``applied`` only AFTER its unmodified step returns.
The network sends these acknowledgements at chunk boundaries, not every step.
Finished rows retain their original slot; they can never be renumbered/revived.
"""
from copy import deepcopy


class AppliedBatchClock:
    def __init__(self, count, chunk_size=16):
        if type(count) is not int or not 1 <= count <= 4 or chunk_size != 16:
            raise ValueError('Registered one-to-four environments and chunk16 required')
        self.count, self.chunk_size = count, chunk_size
        self.steps = [0] * count
        self.active = set(range(count))
        self.pending = {}
        self.issued = None

    def offer(self, tokens):
        if self.pending or self.issued is not None or set(tokens) != self.active:
            raise ValueError('No new chunks before all prior ACKs have been drained')
        if len(set(tokens.values())) != len(tokens) or any(not isinstance(v, str) or not v for v in tokens.values()):
            raise ValueError('Each environment needs its own nonempty action token')
        self.pending = {i: dict(token=t, controls=0, start=self.steps[i]) for i, t in tokens.items()}

    def select_active(self, active_indices):
        ids = list(active_indices)
        current = set(ids)
        if (len(ids) != len(current) or any(type(i) is not int for i in ids)
                or not current <= self.active or not current or self.issued is not None):
            raise ValueError('Duplicated, revived, foreign or unacknowledged active slots')
        # Excluding an already finished row changes only scheduling. Never use
        # this mask to construct an actor feature or a success/failure label.
        self.active = current

    def before_step(self, active_indices):
        self.select_active(active_indices)
        self.issued = tuple(active_indices)

    def applied(self, active_indices):
        ids = tuple(active_indices)
        if ids != self.issued or not all(i in self.pending for i in ids):
            raise ValueError('ACK differs from the actually issued active actions')
        if any(self.pending[i]['controls'] >= self.chunk_size for i in ids):
            raise ValueError('Executed more actions than the offered open-loop chunk')
        for i in ids:
            self.pending[i]['controls'] += 1
            self.steps[i] += 1
        self.issued = None

    def needs_chunk(self):
        return not self.pending or any(self.pending[i]['controls'] == self.chunk_size for i in self.active)

    def drain(self):
        # Called before issuing this step, OR on terminal close. A failed
        # simulator step never fabricates an acknowledgement for that action.
        if self.issued is not None:
            raise ValueError('Cannot drain while an action awaits simulator return')
        result = deepcopy(self.pending)
        if any(row['controls'] <= 0 for row in result.values()):
            raise ValueError('An offered but unexecuted chunk cannot be acknowledged')
        self.pending = {}
        return result


def validate_episode_metadata(metadata, *, allowed_mode):
    if (allowed_mode not in ('train_smoke', 'public_test') or not isinstance(metadata, list)
            or not 1 <= len(metadata) <= 4):
        raise ValueError('Explicit bounded batch identities required')
    seen = set()
    for row in metadata:
        if set(row) != {'task', 'instance_id', 'split', 'episode'}:
            raise ValueError('No privileged fields in episode metadata')
        if (row['split'] != allowed_mode or type(row['instance_id']) is not int
                or not isinstance(row['task'], str) or not row['task'].strip()
                or not isinstance(row['episode'], str) or not row['episode']):
            raise ValueError('Invalid episode identity')
        if allowed_mode == 'public_test' and not 301 <= row['instance_id'] <= 310:
            raise ValueError('Only the registered first ten public instances')
        key = (row['task'], row['instance_id'], row['episode'])
        if key in seen:
            raise ValueError('Duplicate episode')
        seen.add(key)
    if len({r['task'] for r in metadata}) != 1 or len({r['episode'] for r in metadata}) != len(metadata):
        raise ValueError('Mixed-task batch or aliased episode token')
    if len({r['instance_id'] for r in metadata}) != len(metadata):
        raise ValueError('Repeated instance in one official batch')
    return deepcopy(metadata)

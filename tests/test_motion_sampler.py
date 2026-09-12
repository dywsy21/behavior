from collections import Counter
from copy import deepcopy
import itertools
from types import SimpleNamespace

import pytest

from g05.utils.common.motion_sampler import ResumableDistributedMotionBatchSampler
from g05.utils.common.task_event_sampler import branch_fingerprint


class View:
    is_training_set = True
    is_rank_sharded = use_weight_for_sampling = False
    dataset_dirs = ["/motion_test"]
    _memlite_sidecar = SimpleNamespace(path="/motion_sidecar")
    _active_episode_indices = [0, 1, 2, 3, 4]

    def __init__(self):
        self.lookup, self.tasks = {}, []
        cursor = 0
        for task in range(5):
            t = {"task_id": task, "low_strata": {}}
            for kind in ("high_original", "high_recovery", "yaw_stop", "yaw_start_reverse", "gripper", "recovery", "other"):
                n = 40 if kind != "other" else 2000
                for i in range(cursor, cursor+n):
                    self.lookup[i] = (task, kind)
                if kind.startswith("high"):
                    t[kind] = list(range(cursor, cursor+n))
                else:
                    t["low_strata"][kind] = [[cursor, cursor+n]]
                cursor += n
            t["high"] = t["high_original"] + t["high_recovery"]
            self.tasks.append(t)
        self.size = cursor

    def __len__(self):
        return self.size

    def get_memlite_branch_sampling_spec(self):
        return {"high": [i for t in self.tasks for i in t["high"]],
                "low_ranges": [r for t in self.tasks for rs in t["low_strata"].values() for r in rs]}

    def index(self):
        return {"schema_version": 2, "dataset_length": len(self),
            "branch_fingerprint": branch_fingerprint(self.get_memlite_branch_sampling_spec(), len(self)),
            "view_identity": {"dataset_roots": [self.dataset_dirs], "sidecar_paths": [self._memlite_sidecar.path],
                              "active_episodes": [self._active_episode_indices]},
            "low_quotas": {"yaw_stop": 1, "yaw_start_reverse": 1, "gripper": 1, "recovery": 1, "other": 3},
            "high_recovery_period": 4, "tasks": deepcopy(self.tasks)}


def sampler(v, rank=0, index=None):
    return ResumableDistributedMotionBatchSampler(v, sampling_index=index or v.index(),
        batch_size=8, num_replicas=4, rank=rank, seed=17, high_fraction=.125)


def test_quotas_task_equality_and_permanent_rank_disjointness_under_repetition():
    v = View()
    counts, seen = Counter(), []
    for rank in range(4):
        owned = set()
        for n, batch in enumerate(itertools.islice(sampler(v, rank), 200)):
            c = Counter(v.lookup[i][1] for i in batch)
            assert c["yaw_stop"] == c["yaw_start_reverse"] == c["gripper"] == c["recovery"] == 1
            assert c["other"] == 3
            assert c["high_recovery"] == (n % 4 == 3)
            owned.update(batch)
            counts.update(v.lookup[i] for i in batch)
        seen.append(owned)
    for a in range(4):
        for b in range(a):
            assert seen[a].isdisjoint(seen[b])
    for task in range(5):
        assert counts[task, "yaw_stop"] == 160
        assert counts[task, "high_original"] == 120
        assert counts[task, "high_recovery"] == 40


def test_motion_resume_exact_and_bad_partition_rejected():
    v = View()
    s = sampler(v)
    first = list(itertools.islice(s, 100))
    s.set_start_batch(37)
    assert list(itertools.islice(s, 63)) == first[37:]
    bad = v.index()
    bad["tasks"][0]["low_strata"]["recovery"] = bad["tasks"][0]["low_strata"]["yaw_stop"]
    with pytest.raises(ValueError, match="Overlapping"):
        sampler(v, index=bad)

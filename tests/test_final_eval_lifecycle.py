import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval/memlite_final'))
from lifecycle import AppliedBatchClock, validate_episode_metadata, episode_portions


def test_ack_real_partial_and_survivor_stays_in_slot():
    c = AppliedBatchClock(2)
    c.offer({0:'episode-a-chunk0', 1:'episode-b-chunk0'})
    for step in range(16):
        ids = [0, 1] if step < 5 else [1]
        c.before_step(ids)
        c.applied(ids)
    assert c.steps == [5, 16]
    assert c.needs_chunk()
    assert c.drain() == {0:dict(token='episode-a-chunk0', controls=5, start=0),
                         1:dict(token='episode-b-chunk0', controls=16, start=0)}
    c.offer({1:'episode-b-chunk1'})
    with pytest.raises(ValueError, match='revived'):
        c.before_step([0, 1])
    c.before_step([1]); c.applied([1])
    assert c.steps == [5, 17]
    assert c.drain()[1]['controls'] == 1


def test_failed_step_has_no_fabricated_ack():
    c = AppliedBatchClock(1); c.offer({0:'a'}); c.before_step([0])
    assert c.steps == [0]
    with pytest.raises(ValueError): c.drain()
    with pytest.raises(ValueError): c.applied([])
    assert c.steps == [0]


def test_duplicate_ack_or_overexecution_rejected():
    c = AppliedBatchClock(1); c.offer({0:'a'})
    for _ in range(16): c.before_step([0]); c.applied([0])
    with pytest.raises(ValueError): c.applied([0])
    c.before_step([0])
    with pytest.raises(ValueError): c.applied([0])
    assert c.steps == [16]


def test_metadata_no_cross_task_or_truth_or_reused_instance():
    a = dict(task='turning_on_radio', instance_id=301, split='public_test', episode='run-a-301')
    b = dict(a, instance_id=302, episode='run-a-302')
    assert validate_episode_metadata([a,b], allowed_mode='public_test') == [a,b]
    for bad in (dict(b, task='wash_dog_toys'), dict(b, success=True),
                dict(b, episode=a['episode']), dict(b, instance_id=301), dict(b, instance_id=311)):
        with pytest.raises(ValueError): validate_episode_metadata([a,bad], allowed_mode='public_test')


def test_fresh_batch_is_independent():
    a, b = AppliedBatchClock(2), AppliedBatchClock(2)
    a.offer({0:'a',1:'b'}); a.before_step([0,1]); a.applied([0,1])
    assert b.steps == [0,0] and b.pending == {}


def test_same_train_workload_and_exact_public_coverage_for_both_batch_sizes():
    for size in (2,4):
        assert episode_portions(size,preflight=True)==[(size,[1,2,3,4])]
        parts=episode_portions(size)
        assert [i for count,indices in parts for i in indices]==list(range(10))
        assert all(len(indices)%count==0 for count,indices in parts)

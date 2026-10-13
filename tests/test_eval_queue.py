from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
from queue_champion_after_memlite import dependency_ready


def test_only_full_successful_completion_releases_next_owned_gpu_job():
    assert not dependency_ready(dict(status='running'),dict(completed=999))
    assert dependency_ready(dict(status='completed'),dict(status='complete',completed=1000,missing=0))
    with pytest.raises(ValueError):dependency_ready(dict(status='completed'),dict(status='complete',completed=999,missing=1))


@pytest.mark.parametrize('state',['needs_diagnosis','failed_no_retry','failed'])
def test_failure_is_not_permission_to_reroll_or_take_over(state):
    with pytest.raises(RuntimeError):dependency_ready(dict(status=state),{})

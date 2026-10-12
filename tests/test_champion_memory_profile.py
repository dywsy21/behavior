import json
import pytest

from behavior_champion.memory_profile import ProcessMemoryProfile, process_memory_mib


def test_memory_is_bound_to_pid_and_not_total_gpu():
    assert process_memory_mib('11, 100\n 22, 8192\n', 11) == 100
    assert process_memory_mib('', 11) == 0
    assert process_memory_mib('22, 8192', 11) == 0


@pytest.mark.parametrize('text', ['11, N/A', '11, nan', '11, -1', '11, 1\n11, 2', '11,1,extra'])
def test_unsupported_or_ambiguous_sample_is_not_zero(text):
    with pytest.raises(ValueError):
        process_memory_mib(text, 11)


def test_profile_reports_error_without_inventing_hardware_acceptance(tmp_path, monkeypatch):
    profile = ProcessMemoryProfile(tmp_path, '3', interval=30)
    values = iter(['GPU-test, A800, 81920', f'{profile.pid}, 100', f'{profile.pid}, N/A', f'{profile.pid}, 500'])
    monkeypatch.setattr(profile, '_query', lambda fields: next(values))
    profile.start()
    profile.phase='serving'
    profile.sample()
    profile.record_jax('loaded', dict(bytes_in_use=100, peak_bytes_in_use=200))
    result=profile.finish()
    assert result['sampled_process_peak_mib']==500
    assert result['sample_errors']==1
    assert result['phase_peak_mib']==dict(loading=100, serving=500)
    assert result['not_24gb_hardware_acceptance'] is True
    assert result['sampled_peak_is_lower_bound'] is True
    rows=[json.loads(line) for line in (tmp_path/'gpu-memory-samples.jsonl').read_text().splitlines()]
    assert len(rows)==3 and 'error' in rows[1] and 'process_mib' not in rows[1]

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('frozen_transport', Path(__file__).resolve().parents[1] / 'scripts/infra/transfer_frozen_eval_assets.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def fixture(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    files = []
    for name in ('low.pt', 'observer_backbone.pt'):
        data = (name.encode() * 31) + bytes(range(255))
        (source / name).write_bytes(data)
        files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (source / 'manifest.json').write_text(json.dumps(dict(files=files)))
    plan_path = tmp_path / 'plan.json'
    t.prepare(source, t.digest(source / 'manifest.json'), plan_path, chunk_bytes=137)
    return source, json.loads(plan_path.read_text())


def test_ranges_roundtrip_exact_and_never_overwrite(tmp_path):
    source, plan = fixture(tmp_path)
    parts = tmp_path / 'parts'; target = tmp_path / 'target'; target.mkdir()
    (target / 'manifest.json').write_bytes((source / 'manifest.json').read_bytes())
    for row in plan['files']:
        original = (source / row['name']).read_bytes()
        for chunk in row['chunks']:
            data = original[chunk['offset']:chunk['offset'] + chunk['bytes']]
            t.receive(parts, chunk['name'], len(data), chunk['sha256'], io.BytesIO(data))
            with pytest.raises(FileExistsError):
                t.receive(parts, chunk['name'], len(data), chunk['sha256'], io.BytesIO(data))
    result = t.assemble(plan, parts, target)
    assert result['status'] == 'exact_original_files_published'
    for row in plan['files']:
        assert (source / row['name']).read_bytes() == (target / row['name']).read_bytes()
    with pytest.raises(FileExistsError):
        t.assemble(plan, parts, target)


@pytest.mark.parametrize('change', ['truncated', 'extra', 'wrong_hash'])
def test_bad_stream_never_publishes(tmp_path, change):
    data = b'abcdef'; sha = hashlib.sha256(data).hexdigest()
    data = data[:-1] if change == 'truncated' else data + b'z' if change == 'extra' else b'ghijkl'
    with pytest.raises(ValueError):
        t.receive(tmp_path, 'low-000.part', 6, sha, io.BytesIO(data))
    assert not (tmp_path / 'low-000.part').exists()


def test_overlap_changed_source_and_wrong_pack_rejected(tmp_path):
    source, plan = fixture(tmp_path)
    plan['files'][0]['chunks'][1]['offset'] += 1
    with pytest.raises(ValueError):
        t.check_plan(plan)
    (source / 'low.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError):
        t.prepare(source, t.digest(source / 'manifest.json'), tmp_path / 'bad.json')
    with pytest.raises(ValueError):
        t.receive(tmp_path, '../escape', 6, '0' * 64, io.BytesIO(b'abcdef'))


def test_explicit_repair_still_uses_complete_signed_plan(tmp_path):
    _,plan=fixture(tmp_path)
    rows=t.selected_parts(plan,['observer_backbone-001.part'])
    assert len(rows)==1 and rows[0][1]['offset']==137
    assert len(t.selected_parts(plan))>1
    for bad in ([],['not-registered.part'],['low-000.part','low-000.part']):
        with pytest.raises(ValueError):t.selected_parts(plan,bad)

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
import champion_fifo_eval as m


def test_only_admission_and_worker_dispatch_change_not_actor_functions(tmp_path):
    entry=tmp_path/'scripts/eval/champion_full_eval.py'
    worker,supervise,model=object(),object(),object()
    runtime=SimpleNamespace(REPO=tmp_path,HERE=entry.parent,__file__=str(entry),
        loader_lock=object(),worker=worker,supervise=supervise,model=model)
    receipt=m.bind_admission(runtime,tmp_path)
    assert runtime.loader_lock is m.loader_lock and runtime.__file__==m.__file__
    assert runtime.REPO==tmp_path and runtime.HERE==entry.parent
    assert (runtime.worker,runtime.supervise,runtime.model)==(worker,supervise,model)
    assert receipt['original_dispatcher']==str(entry)


@pytest.mark.parametrize('field',['REPO','HERE','__file__'])
def test_wrong_actor_paths_are_rejected_before_any_binding(tmp_path,field):
    entry=tmp_path/'scripts/eval/champion_full_eval.py';old=object()
    runtime=SimpleNamespace(REPO=tmp_path,HERE=entry.parent,__file__=str(entry),loader_lock=old)
    setattr(runtime,field,tmp_path/'foreign')
    with pytest.raises(ValueError):m.bind_admission(runtime,tmp_path)
    assert runtime.loader_lock is old

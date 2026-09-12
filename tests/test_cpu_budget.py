import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

target = Path(__file__).with_name("run_with_cpu_budget.py")
if not target.exists():
    target = Path(__file__).resolve().parents[1] / "scripts/run_with_cpu_budget.py"
spec = importlib.util.spec_from_file_location("cpu_budget", target)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)

@pytest.mark.parametrize("value,expected", [("0-2,5", {0, 1, 2, 5}), ("3", {3}), ("1,1,2", {1, 2})])
def test_affinity(value, expected):
    assert budget.parse_affinity(value) == expected

@pytest.mark.parametrize("value", ["", "-1", "3-1", "a", "1-2-3", "1,"])
def test_bad_affinity(value):
    with pytest.raises(ValueError):
        budget.parse_affinity(value)

@pytest.mark.parametrize("value", ["0", "-1", "1025", "x"])
def test_bad_threads(value):
    with pytest.raises((ValueError, budget.argparse.ArgumentTypeError)):
        budget.positive_int(value)

def test_configures_before_import_and_preserves_gpu(monkeypatch):
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OPENCV_FOR_THREADS_NUM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    state = {}
    torch = SimpleNamespace(set_num_threads=lambda x: state.update(torch=x),
        set_num_interop_threads=lambda x: state.update(interop=x),
        get_num_threads=lambda: state["torch"], get_num_interop_threads=lambda: state["interop"])
    cv2 = SimpleNamespace(setNumThreads=lambda x: state.update(cv2=x), getNumThreads=lambda: state["cv2"])
    def get_module(name):
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "2"
        return {"torch": torch, "cv2": cv2}[name]
    monkeypatch.setattr(budget.importlib, "import_module", get_module)
    monkeypatch.setattr(budget.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    result = budget.configure(2, 1, 1)
    assert result["torch_threads"] == 2 and result["interop_threads"] == 1
    assert result["opencv_threads"] == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

def test_rejects_foreign_affinity_before_mutation(monkeypatch):
    monkeypatch.setattr(budget.os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(budget.os, "sched_setaffinity", lambda *a: pytest.fail("must not mutate affinity"))
    with pytest.raises(ValueError, match="unavailable"):
        budget.configure(1, 1, 1, "2-3")

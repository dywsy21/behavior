import importlib.util
from pathlib import Path
import pytest

target = Path(__file__).with_name("a100_eval_profile.py")
if not target.exists():
    target = Path(__file__).resolve().parents[1] / "scripts/a100_eval_profile.py"
spec = importlib.util.spec_from_file_location("a100_profile", target)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GPU_CSV = "\n".join(f"{i}, NVIDIA A100 80GB PCIe, 81920" for i in range(4))
TOPO = "GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID\n" + "\n".join(
    f"GPU{i} X SYS SYS SYS {i*24}-{i*24+23} {i} N/A" for i in range(4))

def test_real_shape():
    result = mod.parse_profile(GPU_CSV, TOPO, set(range(96)))
    assert [r["cpu_affinity"] for r in result["gpus"]] == ["0-23", "24-47", "48-71", "72-95"]
    assert result["physics"] == "stock_cpu_unchanged"

@pytest.mark.parametrize("gpu,topo,cpus", [
    (GPU_CSV.replace("A100", "H100"), TOPO, set(range(96))),
    (GPU_CSV.replace("81920", "40960"), TOPO, set(range(96))),
    (GPU_CSV.splitlines()[0], TOPO, set(range(96))),
    (GPU_CSV, TOPO.replace("24-47", "0-23"), set(range(96))),
    (GPU_CSV, TOPO, set(range(48))),
    (GPU_CSV, "", set(range(96))),
])
def test_rejects_wrong_machine_or_affinity(gpu, topo, cpus):
    with pytest.raises(ValueError):
        mod.parse_profile(gpu, topo, cpus)

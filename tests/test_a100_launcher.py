"""Exercise the real launcher functions with harmless executable stand-ins."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import pytest

LAUNCHER = Path(__file__).with_name("eval_memlite_v9.sh")
if not LAUNCHER.exists():
    LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/eval_memlite_v9.sh"

def function(name):
    source = LAUNCHER.read_text()
    found = re.search(r"^" + name + r"\(\) \{\n.*?^\}", source, re.M | re.S)
    assert found, name
    return found[0]

@pytest.mark.parametrize("fast,video,expected_threads", [(0, 0, None), (1, 1, "1"), (1, 0, "1")])
def test_eval_command_and_video(tmp_path, fast, video, expected_threads):
    run = tmp_path / "run"
    mock = tmp_path / "mock.py"
    mock.write_text("#!/usr/bin/env python3\nimport json,sys,pathlib\na=sys.argv[1:]\n"
                    "print(json.dumps(a))\np=pathlib.Path(a[a.index('--output-dir')+1])/'json'\n"
                    "p.mkdir(parents=True)\n(p/'result.json').write_text('{\"success\":false,\"steps\":1}')\n")
    mock.chmod(0o755)
    script = "\n".join([
        "set -Eeuo pipefail", f"RUN_ROOT={shlex.quote(str(run))}",
        f"OG_ROOT={shlex.quote(str(tmp_path))}", f"OG_PYTHON={shlex.quote(str(mock))}",
        "CPU_BUDGET_RUNNER=/budget.py", "EVAL_RUNNER=/runner.py", "ROBOT_CONFIG=/r1pro.yaml", "GLU_LIB=/glu",
        f"A100_PROFILE={fast}", f"WRITE_VIDEO={video}", "CPU_AFFINITIES=(0-23 24-47 48-71 72-95)",
        "log() { :; }", function("run_eval"), "run_eval synthetic 2 8767"])
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    arguments = json.loads((run / "tasks/synthetic/eval.log").read_text())
    assert ("--write-video" in arguments) == bool(video)
    assert ("--no-write-video" in arguments) == (not video)
    assert arguments[arguments.index("--instance-indices") + 1] == "0"
    assert "--max-steps" not in arguments
    assert ("--threads" in arguments) == bool(fast)
    if fast:
        assert arguments[arguments.index("--threads") + 1] == expected_threads
        assert arguments[arguments.index("--cpu-affinity") + 1] == "48-71"
    assert (run / "tasks/synthetic/exit_code.txt").read_text().strip() == "0"

@pytest.mark.parametrize("fast", [0, 1])
def test_policy_command_keeps_ar_contract(tmp_path, fast):
    repo = tmp_path / "repo"
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    script = "\n".join([
        "set -Eeuo pipefail", f"RUN_ROOT={shlex.quote(str(run))}", f"REPO={shlex.quote(str(repo))}",
        "CPU_BUDGET_RUNNER=/budget.py", "MEM_SERVER=/serve_behavior_policy_mem.py", "TASKS=/tasks.jsonl", "CHECKPOINT=/step_5000.pt",
        "REPLAN_EVERY_CHUNKS=8", "MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS=1024", "server_pids=()",
        f"A100_PROFILE={fast}", "CPU_AFFINITIES=(0-23 24-47 48-71 72-95)",
        function("start_policy_server"), 'start_policy_server 3 8768', 'wait "${server_pids[0]}"'])
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    args = json.loads((run / "logs/mem_policy_gpu3_port8768.log").read_text())
    assert all(x in args for x in ("--pure-ar", "--require-memlite", "model.model_weights_to_bf16=true"))
    assert args[args.index("--action_steps") + 1] == "16"
    assert args[args.index("--memlite-replan-every-chunks") + 1] == "8"
    if fast:
        assert args[args.index("--threads") + 1] == "2"
        assert args[args.index("--cpu-affinity") + 1] == "72-95"
    else:
        assert "--threads" not in args

@pytest.mark.parametrize("tag", ["../../x", "a/b", "x y", ";id", "x" * 81])
def test_rejects_unsafe_run_tag_before_any_work(tag):
    result = subprocess.run(["bash", str(LAUNCHER), "--train-output", "/unused", "--run-tag", tag],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "safe identifier" in result.stderr

def test_gpu_caches_only_and_idempotent(tmp_path):
    run, cache = tmp_path / "run", tmp_path / "cache"
    script = "\n".join(["set -Eeuo pipefail", f"RUN_ROOT={shlex.quote(str(run))}",
        f"SHARED_A100_CACHE={shlex.quote(str(cache))}", function("prepare_a100_cache"),
        "prepare_a100_cache", "prepare_a100_cache"])
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    links = list((run / "appdata").glob("gpu*/*/cache"))
    assert len(links) == 8 and all(p.is_symlink() for p in links)
    assert all(p.resolve().is_relative_to(cache) for p in links)
    assert not list((run / "appdata").glob("gpu*/*/data"))

def test_refuses_foreign_cache_link(tmp_path):
    run, cache = tmp_path / "run", tmp_path / "cache"
    link = run / "appdata/gpu0/local/cache"
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "unrelated", target_is_directory=True)
    script = "\n".join(["set -Eeuo pipefail", f"RUN_ROOT={shlex.quote(str(run))}",
        f"SHARED_A100_CACHE={shlex.quote(str(cache))}", function("prepare_a100_cache"), "prepare_a100_cache"])
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 2
    assert link.readlink() == tmp_path / "unrelated"

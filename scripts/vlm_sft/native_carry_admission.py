"""Explicit partial inheritance; old engineering gates never validate new duration."""
import json
from pathlib import Path
from common import sha
from native_actor_protocol import check_sha
from native_execution import CARRY_PROFILE,authorization_profile

BASE_EXECUTOR="1fcc505cb570c0fa215c408b2ae78d904819dc95b40bbbe117a560106ad97f4c"
BASE_CODE="13bd4bb716b386bc7624d05d03f38e77286470ae"
BASE_GATES={0:"beb3de5451ba8e73e995b102245bcece214dc462af521093903fdba06ac0cea0",
            3:"1e3924e08c27d34129bad90e8fe21ad32535bee3f60e86283177cc63f8fbdfa1"}
INHERITANCE={"mode":"original-1fcc-gates-unmodified-scope-only",
    "base_executor_digest":BASE_EXECUTOR,"new_carry_duration_physically_validated_by_old_gates":False}
BOOTSTRAP_NAME="native_t1_i192_p0388_ws45_cd1_b2"


def review_file(value):
    if not isinstance(value,dict) or set(value)!={"path","sha256"}:raise ValueError("Exact parent review path/SHA required")
    check_sha(value["sha256"]);path=Path(value["path"])
    if not path.is_absolute() or sha(path)!=value["sha256"]:raise ValueError("Parent review bytes changed")
    review=json.loads(path.read_text())
    if review.get("reviewer")!="Codex-parent" or review.get("review_passed") is not True:
        raise ValueError("Independent parent review required")
    return review


def require_carry_reviews(auth,*,bootstrap=False):
    if authorization_profile(auth)!=CARRY_PROFILE:return
    # Bootstrap exception is only for the first exact 388 collection, never
    # training/service/evaluation or a self-signed completed integration gate.
    is_bootstrap=auth.get("carry_duration_bootstrap")
    if type(is_bootstrap) is not bool or is_bootstrap is not bootstrap:
        raise ValueError("Explicit, phase-correct carry bootstrap required")
    if bootstrap and (auth.get("source")!=[1,310,192] or auth.get("paid_prefix_controls")!=388 or
                      auth.get("authorize_offline_teacher") is not True):
        raise ValueError("Only registered first 388 collection may bootstrap")
    executor=auth.get("executor_digest");check_sha(executor)
    core=auth.get("carry_duration_core_commit")
    if not isinstance(core,str) or len(core)!=40 or any(c not in '0123456789abcdef' for c in core):
        raise ValueError("Frozen carry core commit required")
    cpu=review_file(auth.get("carry_duration_cpu_review"))
    if (cpu.get("executor_digest")!=executor or cpu.get("core_commit")!=core or
            cpu.get("carry_duration_v1") is not True or cpu.get("original_safety_and_rate_limits_unchanged") is not True):
        raise ValueError("New-core CPU review must bind unchanged safety/rate gates")
    integration=auth.get("carry_duration_integration_review")
    if bootstrap:
        if integration is not None:raise ValueError("Bootstrap cannot claim an already completed physical gate")
        return
    actual=review_file(integration)
    ticks=actual.get("completed_extended_carry_macro_ticks")
    if (actual.get("run_name")!=BOOTSTRAP_NAME or actual.get("executor_digest")!=executor or
            actual.get("core_commit")!=core or authorization_profile(actual)!=CARRY_PROFILE or
            actual.get("integration_passed") is not True or
            not isinstance(ticks,list) or not ticks or any(type(t) is not int or not 40<t<=75 for t in ticks)):
        raise ValueError("Whole first-388 actual extended-carry integration review required")
    check_sha(actual.get("inventory_sha256"))


def require_engineering_gates(auth,executor,*,bootstrap=False):
    profile=authorization_profile(auth)
    gates=[(Path(p),json.loads(Path(p).read_text())) for p in auth["engineering_gate_paths"]]
    if len(gates)!=2 or {g["task"] for _,g in gates}!={0,3}:raise ValueError("Both engineering gates required")
    if profile==CARRY_PROFILE:
        if auth.get("engineering_gate_inheritance")!=INHERITANCE or executor!=auth.get("executor_digest"):
            raise ValueError("Explicit old-gate partial inheritance, never a new duration claim")
        require_carry_reviews(auth,bootstrap=bootstrap)
        for path,g in gates:
            if (sha(path)!=BASE_GATES[g['task']] or g.get('implementation_digest')!=BASE_EXECUTOR or
                    g.get('carry_duration_v1',False) is not False):
                raise ValueError("Original immutable gate bytes/old mode required")
        expected=BASE_EXECUTOR
    else:
        if any(k in auth for k in ('engineering_gate_inheritance','carry_duration_bootstrap','carry_duration_cpu_review',
                                  'carry_duration_integration_review','carry_duration_core_commit')):
            raise ValueError("Old profiles cannot inherit a new execution gate")
        expected=executor
    if any(g.get("gate_ok") is not True or g.get("robot_geometry_guards") is not True or
           g.get("gripper_completion_v1",False) is not (profile is not None) or
           g.get("implementation_digest")!=expected for _,g in gates):
        raise ValueError("Reviewed engineering gates do not match the declared scope")

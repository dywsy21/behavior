"""POSTHOC scoring only; the public evaluation interface never imports this.

Unspecified-hand GRASP may succeed with either hand. Each hand must separately
satisfy the full unchanged causal predicate: no cross-hand evidence pooling.
"""
from native_teacher_outcomes import LocalOutcome


def score_grasp(spec,initial,frames,issued_tokens,final_hold,*,specified_hand=None):
    if spec.get("verb")!="GRASP" or specified_hand not in (None,"left","right"):
        raise ValueError("Registered GRASP instruction semantics required")
    hands=(specified_hand,) if specified_hand else ("left","right")
    scores={}
    for arm in hands:
        oracle=LocalOutcome({**spec,"hand":arm,"support_hand":None})
        oracle.update(initial)
        for frame in frames:oracle.update(frame,issued_tokens.get(frame["tick"]))
        verdict=oracle.update(final_hold)
        scores[arm]=verdict
    succeeded=[arm for arm,r in scores.items() if r["outcome"]=="SUCCEEDED"]
    return {"per_hand":scores,"actual_successful_hands":succeeded,"any_hand_local_success":bool(succeeded),
        "specified_hand":specified_hand,"posthoc_only":True,"official_task_success_claim":False}

"""H66: CPU-only nonvisual diagnostics on existing TRAIN completion states.

No image inference, new labels, training sidecars, policy deployment or SR.
Group metadata is used ONLY to split; private provenance never is a feature.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


STATES_SHA = "2fd27d707ad86e7fd6c09662000e99af26c0b1d686bbe9ff92b3c6aa8604fa99"
POSITIVE = "REQUEST_VERIFY"
NEGATIVE = "CONTINUE"


def label(row):
    value = row["label"]["skill_status"]
    if value not in (NEGATIVE, POSITIVE): raise ValueError("Known two-class label required")
    return value == POSITIVE


def history_key(row):
    return tuple(row["actor"]["history"])


def trailing_up(row):
    count = 0
    for item in reversed(history_key(row)):
        if item != "RIGHT_UP": break
        count += 1
    return count


def proprio_features(row):
    # Explicit whitelist. No id, task label, image hash, clock, instance,
    # terminal flag, ordinal position, private state or future command.
    proprio = row["actor"]["proprio"]
    joints = proprio["joint_positions_rad"]
    values = [*joints["torso"], *joints["left"], *joints["right"],
              proprio["finger_opening_m"]["left"]/.05, proprio["finger_opening_m"]["right"]/.05]
    features = np.asarray(values, dtype=float)
    if features.shape != (20,) or not np.isfinite(features).all():
        raise ValueError("Finite q18 plus measured apertures required")
    return features


def counts(truth, predicted):
    if not len(truth) or len(truth) != len(predicted): raise ValueError("Nonempty paired decisions required")
    tp = sum(t and p for t, p in zip(truth, predicted))
    tn = sum(not t and not p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    return {"n": len(truth), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": (tp+tn)/len(truth), "request_recall": recall,
            "continue_false_request_rate": fp/(fp+tn) if fp+tn else None,
            "balanced_accuracy": (recall+specificity)/2 if recall is not None and specificity is not None else None}


def majority(rows):
    # Tie deliberately resolves to CONTINUE, not input/id order.
    return sum(label(row) for row in rows) > len(rows)/2


def fit_threshold(rows):
    truth = [label(row) for row in rows]
    if len(set(truth)) != 2: raise ValueError("Both classes in training fold required")
    # Six means never request in this five-token history. Ties prefer the
    # larger threshold: this deterministic choice is declared before evaluation.
    scored = [(counts(truth, [trailing_up(row) >= threshold for row in rows])["balanced_accuracy"], threshold)
              for threshold in range(7)]
    return max(scored)[1]


def fold_predictions(train, test):
    prior = majority(train)
    table = defaultdict(list)
    for row in train: table[history_key(row)].append(row)
    threshold = fit_threshold(train)
    features = np.stack([proprio_features(row) for row in train])
    output = []
    for row in test:
        # Stable original TRAIN row order resolves exact-distance ties.
        distances = np.linalg.norm(features-proprio_features(row), axis=1)
        nearest = train[int(np.argmin(distances))]
        matches = table.get(history_key(row), [])
        output.append({"id": row["id"], "truth": label(row),
            "predictions": {"majority": prior, "history_lookup": majority(matches) if matches else prior,
                            "train_fitted_up_threshold": trailing_up(row) >= threshold,
                            "proprio_1nn": label(nearest)},
            "history_seen_in_train": bool(matches), "trailing_up": trailing_up(row),
            "nearest_train_id": nearest["id"], "nearest_train_group": nearest["provenance"]["group"],
            "nearest_fixed_scale_distance": float(distances.min())})
    return threshold, output


def conflict_buckets(rows, key):
    table = defaultdict(list)
    for row in rows: table[key(row)].append(row)
    return [[row["id"] for row in values] for values in table.values() if len({label(row) for row in values}) > 1]


def analyze(rows):
    if not rows or len({row["id"] for row in rows}) != len(rows): raise ValueError("Unique state ids required")
    groups = sorted({tuple(row["provenance"]["group"]) for row in rows})
    if len(groups) < 2: raise ValueError("At least two distinct source-instance groups required")
    folds, all_predictions = [], []
    for group in groups:
        train = [row for row in rows if tuple(row["provenance"]["group"]) != group]
        test = [row for row in rows if tuple(row["provenance"]["group"]) == group]
        threshold, predictions = fold_predictions(train, test)
        metrics = {name: counts([r["truth"] for r in predictions], [r["predictions"][name] for r in predictions])
                   for name in predictions[0]["predictions"]}
        folds.append({"held_out_TRAIN_group": list(group), "train_ids": [row["id"] for row in train],
                      "train_selected_up_threshold": threshold, "metrics": metrics, "predictions": predictions})
        all_predictions.extend(predictions)
    metrics = {name: counts([r["truth"] for r in all_predictions], [r["predictions"][name] for r in all_predictions])
               for name in all_predictions[0]["predictions"]}
    # Exact available nonvisual prompt content, excluding image hash identity.
    def nonvisual(row):
        return json.dumps({k: row["actor"][k] for k in ("protocol", "task", "active_instruction", "proprio", "history")},
                          sort_keys=True, allow_nan=False)
    return {"status": "TRAIN_ONLY_SHORTCUT_DIAGNOSTIC", "states": len(rows),
            "classes": dict(Counter(row["label"]["skill_status"] for row in rows)),
            "source_instance_groups": [list(g) for g in groups], "folds": folds, "pooled_out_of_group": metrics,
            "same_history_conflicting_label_buckets": conflict_buckets(rows, history_key),
            "identical_nonvisual_input_conflicting_label_buckets": conflict_buckets(rows, nonvisual),
            "neural_training_updates": 0, "model_calls": 0, "simulator_resets": 0,
            "success_rate": None, "not_independent_eval_or_visual_causality": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if dirty: raise ValueError("Freeze audited source before actual H66 statistics")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.output.exists() or args.output.is_symlink(): raise ValueError("New output only")
    if args.states.is_symlink() or not args.states.is_file() or args.states.stat().st_size > 1024**2:
        raise ValueError("Bounded regular input required")
    raw = args.states.read_bytes()
    if hashlib.sha256(raw).hexdigest() != STATES_SHA: raise ValueError("Exact reviewed TRAIN39 input required")
    rows = [json.loads(line) for line in raw.splitlines()]
    from native_completion import validate_sidecar
    for row in rows: validate_sidecar(row)
    if (len(rows) != 39 or Counter(row["label"]["skill_status"] for row in rows) != {NEGATIVE: 34, POSITIVE: 5}
            or {tuple(row["provenance"]["group"]) for row in rows} != {(1, 114), (1, 192)}):
        raise ValueError("Exact original TRAIN groups/counts required")
    result = {**analyze(rows), "source_commit": source, "states_sha256": STATES_SHA,
              "utc": datetime.now(timezone.utc).isoformat(), "seconds": time.monotonic()-started,
              "features": "Only public q18 in radians and average apertures divided by .05; history only in history baselines",
              "limits": "Two similar TRAIN instances only. No images inspected or labels changed. Not a new train split, policy, success-rate trial or dataset release."}
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if len(encoded.encode()) > 1024**2: raise ValueError("Bounded output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream: stream.write(encoded)
    print(json.dumps({"states": len(rows), "seconds": result["seconds"], "metrics": result["pooled_out_of_group"]}))


if __name__ == "__main__": main()

"""Immutable rollout inventory and CAUSAL candidate rows, never auto-approved BC.

No simulator or model import. Physical audit fields are confined to the label
sidecar; actor observations are an explicit allowlist. In particular, audit
at control k describes s[k+1], not the image at s[k].
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from validate_recovery import validate_archive

CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
FINAL_ARCHIVE = re.compile(r"[0-9a-f]{24}\.zip\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def group_key(task, instance):
    # Display spaces vs evaluator underscores do not create a new split group.
    return " ".join(str(task).replace("_", " ").casefold().split()) + ":" + str(int(instance))


def split_group(task, instance, protected=()):
    key = group_key(task, instance)
    if key in protected:
        return "protected"
    # Fixed before window extraction; sibling seeds/policies/branches stay together.
    return "dev" if int(digest(["recovery-v1-20261009", key])[:8], 16) % 5 == 0 else "train"


def read_archive(path):
    if not __debug__:
        raise RuntimeError("The legacy structural verifier requires assertions enabled")
    checked = validate_archive(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        rows = [json.loads(line) for line in archive.read("transitions.jsonl").splitlines()]
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("Unsafe archive member")
    if Path(path).name != manifest["clip_id"] + ".zip":
        raise ValueError("Filename/clip identity mismatch")
    return checked, manifest, rows


def resolve_entity(name, scope, binding):
    """Never guess a simulator entity by class name or a numeric suffix."""
    if not name:
        return None
    if name in scope:
        return name
    candidate = binding.get(name)
    return candidate if candidate in scope else None


def stable_arm(rows_by_step, anchor, entity, arm, count=6, expected="TRUE"):
    # Entries k-count ... k-1 give physical states up to and INCLUDING s[k].
    for step in range(anchor - count, anchor):
        row = rows_by_step.get(step)
        if row is None:
            return False
        states = row["physical_audit"].get("grasp_states", {}).get(entity, {})
        if states.get(arm, "UNKNOWN") != expected:
            return False
    return True


def outcome_candidate(rows_by_step, anchor, skills, binding):
    """Conservative skill-result proposals; all remain unapproved for training."""
    before = rows_by_step.get(anchor - 1)
    if before is None:
        return dict(value=None, reason="missing_pre_observation_audit", members=[])
    scope = before["physical_audit"].get("grasp_states", {})
    members = []
    for skill in skills:
        entity = resolve_entity(skill.get("target"), scope, binding)
        item = dict(verb=skill["verb"], target=skill.get("target"), entity=entity,
                    value=None, reason="unsupported_skill_result_contract")
        if entity is None:
            item["reason"] = "target_binding_missing"
        elif skill["verb"] == "GRASP":
            requested_arm = skill.get("arm", "UNSPECIFIED").upper()
            arms = ([requested_arm.lower()] if requested_arm in {"LEFT", "RIGHT"}
                    else ["left", "right"] if requested_arm == "UNSPECIFIED" else [])
            held = [arm for arm in arms if stable_arm(rows_by_step, anchor, entity, arm)]
            if held:
                item.update(value="SUCCEEDED", reason="same_target_same_arm_6_controls",
                            evidence_arms=held)
            else:
                item["reason"] = "no_stable_grasp_not_a_failure_label"
        members.append(item)
    success = bool(members) and all(m["value"] == "SUCCEEDED" for m in members)
    return dict(value="SUCCEEDED" if success else None,
                reason="all_members_locally_satisfied" if success else "unverified_or_partial_bundle",
                members=members, evidence_end_control_step=anchor)


def anchor_candidate(episode, rows_by_step, anchor, image_ref, binding, split):
    row = rows_by_step[anchor]
    skills = json.loads(row["context"]["active_skills_semantic_json"])
    semantic = canonical(skills)
    future = [rows_by_step.get(step) for step in range(anchor, anchor + 32)]
    continuous = all(r is not None for r in future)
    same_intent = continuous and all(canonical(json.loads(r["context"]["active_skills_semantic_json"]))
                                    == semantic and r["context"]["parent_goal"] == row["context"]["parent_goal"]
                                    for r in future)
    no_early_terminal = continuous and not any(r["terminated"] or r["truncated"] for r in future[:-1])
    valid_actions = bool(continuous and same_intent and no_early_terminal)
    cursor = anchor - 1
    while cursor in rows_by_step:
        previous = rows_by_step[cursor]
        if (canonical(json.loads(previous["context"]["active_skills_semantic_json"])) != semantic
                or previous["context"]["parent_goal"] != row["context"]["parent_goal"]):
            break
        cursor -= 1
    before = rows_by_step.get(anchor - 1)
    physically_held = {}
    if before is not None:
        for entity, arms in before["physical_audit"].get("grasp_states", {}).items():
            held = [arm for arm in ("left", "right") if arm in arms
                    and stable_arm(rows_by_step, anchor, entity, arm)]
            if held:
                physically_held[entity] = held
    episode_identity = [episode["run"], episode["episode_id"]]
    return dict(schema="recovery_anchor_candidate_v1", sample_id=digest([episode_identity, anchor]),
        source_episode=episode_identity, task=episode["task"], instance_id=episode["instance_id"],
        split=split, source_group=group_key(episode["task"], episode["instance_id"]),
        control_step=anchor, policy_update=row["policy_update"],
        actor_input=dict(rgb=image_ref, proprio_before=row["proprio_before"],
                         issued_skills_semantic_json=semantic, parent_goal=row["context"]["parent_goal"],
                         observed_same_intent_controls=anchor - cursor - 1,
                         history_is_partial=cursor not in rows_by_step),
        label_audit=dict(outcome=outcome_candidate(rows_by_step, anchor, skills, binding),
                         physically_held_at_observation=physically_held,
                         full_executed_32_step_target_available=valid_actions,
                         target_availability_is_not_action_quality=True),
        eligibility=dict(outcome=False, planner=False, action=False),
        label_status="candidate_pending_semantic_review", review_status="pending")


def load_indexes(root):
    indexes, pins = {}, {}
    for path in sorted(root.glob("gpu_*/index.jsonl")):
        raw = path.read_bytes()
        pins[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            row = json.loads(line)
            name = Path(row["path"]).name
            key = path.parent.name + "/" + name
            if key in indexes:
                raise ValueError("Duplicate published index entry: " + key)
            indexes[key] = row
    return indexes, pins


def validate_source_episode(episode, source_manifest, task_contracts):
    contract = task_contracts[episode["task"]]
    if (episode["instance_id"] not in contract["train_instances"]
            or episode["source_commit"] != source_manifest["source_commit"]
            or episode["split"].lower() != "train"):
        raise ValueError("Episode is outside the pinned source TRAIN contract")
    for component in ("high", "low"):
        parents = [value for key, value in source_manifest["model"]["checkpoints"].items()
                   if key.startswith(component + "/")]
        if len(parents) != 1 or episode[component + "_checkpoint_sha256"] != parents[0]["sha256"]:
            raise ValueError("Candidate parent checkpoint differs from source manifest")


def inspect_corpus(root, *, protected=(), bindings=None, maximum_archives=512, source_manifest=None):
    root = Path(root).resolve()
    files = sorted(p for p in root.glob("gpu_*/*.zip") if FINAL_ARCHIVE.fullmatch(p.name))
    if not files or len(files) > maximum_archives:
        raise ValueError("Empty or over-budget candidate corpus")
    indexes, index_pins = load_indexes(root)
    task_contracts = ({task["task"]: task for group in source_manifest["groups"] for task in group["tasks"]}
                      if source_manifest else {})
    protected = set(protected)
    for task, contract in task_contracts.items():
        protected.update(group_key(task, i) for i in contract["sft_heldout_excluded"])
        protected.update(group_key(task, i) for i in contract["development_instances"])
    episodes, inventory, quarantined = defaultdict(list), [], []
    for path in files:
        relative = str(path.relative_to(root))
        try:
            checked, manifest, _rows = read_archive(path)
            indexed = indexes[relative]
            if (checked["sha256"] != indexed["sha256"] or path.stat().st_size != indexed["bytes"]
                    or manifest["episode"] != indexed["episode"]):
                raise ValueError("Published index hash/size/episode mismatch")
            if source_manifest:
                validate_source_episode(manifest["episode"], source_manifest, task_contracts)
            entry = dict(path=relative, sha256=checked["sha256"], bytes=path.stat().st_size,
                         episode=manifest["episode"], controls=checked["controls"], images=checked["images"],
                         events=manifest["events"], structural_validation="passed",
                         split=split_group(manifest["episode"]["task"], manifest["episode"]["instance_id"], protected))
            inventory.append(entry)
            ep_key = (manifest["episode"]["run"], manifest["episode"]["episode_id"])
            episodes[ep_key].append((path, manifest))
        except (ValueError, KeyError, AssertionError, OSError, zipfile.BadZipFile) as error:
            quarantined.append(dict(path=relative, error=type(error).__name__ + ": " + str(error)))
    anchors, collisions, overlaps = [], [], 0
    for ep_key, clips in sorted(episodes.items()):
        merged, images = {}, {}
        invalid = False
        for path, manifest in clips:
            if canonical(manifest["episode"]) != canonical(clips[0][1]["episode"]):
                collisions.append(dict(episode=list(ep_key), kind="episode_identity_conflict"))
                invalid = True
            with zipfile.ZipFile(path) as archive:
                rows = [json.loads(line) for line in archive.read("transitions.jsonl").splitlines()]
            for row in rows:
                step = row["control_step"]
                if step in merged:
                    overlaps += 1
                    if canonical(merged[step]) != canonical(row):
                        collisions.append(dict(episode=list(ep_key), step=step, kind="transition_conflict"))
                        invalid = True
                else:
                    merged[step] = row
            for key, meta in manifest["rgb_anchors"].items():
                step = int(key)
                if step in images and images[step]["sha256"] != meta["sha256"]:
                    collisions.append(dict(episode=list(ep_key), step=step, kind="image_conflict"))
                    invalid = True
                images.setdefault(step, dict(archive=str(path.relative_to(root)),
                                             sha256=meta["sha256"], control_step=step))
        if invalid:
            continue
        episode = clips[0][1]["episode"]
        split = split_group(episode["task"], episode["instance_id"], protected)
        binding = (bindings or {}).get(canonical(list(ep_key)), {})
        for anchor in sorted(images):
            if anchor not in merged or merged[anchor]["rgb_anchor_control_step"] != anchor:
                continue  # A ring buffer may begin AFTER its first image anchor.
            anchors.append(anchor_candidate(episode, merged, anchor, images[anchor], binding, split))
    summary = dict(schema="recovery_corpus_audit_v1", archives_seen=len(files),
        structural_passed=len(inventory), quarantined=quarantined,
        bytes=sum(x["bytes"] for x in inventory), episodes=len(episodes),
        tasks=len({x["episode"]["task"] for x in inventory}),
        unique_visual_anchors=len(anchors), duplicated_controls_removed=overlaps,
        conflicting_evidence=collisions, split_counts=dict(Counter(x["split"] for x in anchors)),
        verb_counts=dict(Counter(s["verb"] for a in anchors
                                 for s in json.loads(a["actor_input"]["issued_skills_semantic_json"]))),
        member_reason_counts=dict(Counter(m["reason"] for a in anchors for m in a["label_audit"]["outcome"]["members"])),
        proposed_outcome_counts=dict(Counter(a["label_audit"]["outcome"]["value"] or "UNLABELED" for a in anchors)),
        anchors_with_any_stable_hold=sum(bool(a["label_audit"]["physically_held_at_observation"]) for a in anchors),
        complete_32_step_targets=sum(a["label_audit"]["full_executed_32_step_target_available"] for a in anchors),
        inventory_sha256=digest(inventory), index_sha256=index_pins,
        protected_source_groups=len(protected), source_manifest_supplied=source_manifest is not None,
        approved_training_rows=0, training_ready=False,
        constraints=["No automatic expert/BC/recovery labels", "Physics never in actor_input",
                     "Protected split and semantic/media approval required before any release"])
    return inventory, anchors, summary


def select_review(inventory, count=60):
    """Round-robin task coverage, event diversity, then deterministic clip hash."""
    groups = defaultdict(list)
    for item in inventory:
        groups[item["episode"]["task"]].append(item)
    for task in groups:
        groups[task].sort(key=lambda x: (tuple(sorted(e["kind"] for e in x["events"])), x["sha256"]))
    selected = []
    for depth in range(max((len(v) for v in groups.values()), default=0)):
        for task in sorted(groups):
            if depth < len(groups[task]):
                selected.append(groups[task][depth])
                if len(selected) >= count:
                    return selected
    return selected

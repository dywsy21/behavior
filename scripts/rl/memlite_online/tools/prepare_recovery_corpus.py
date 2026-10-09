"""CPU inventory of already-published RGB/action candidates; no training release."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recovery_corpus import canonical, file_sha, inspect_corpus, select_review  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected-groups", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--max-archives", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new versioned output, do not overwrite an audit")
    protected = json.loads(args.protected_groups.read_text())["groups"] if args.protected_groups else []
    source_manifest = json.loads(args.source_manifest.read_text()) if args.source_manifest else None
    bindings = json.loads(args.bindings.read_text()) if args.bindings else None
    if bindings and (bindings.get("schema") != "recovery_physical_bindings_v1"
                     or args.source_manifest is None
                     or bindings["source_manifest_sha256"] != file_sha(args.source_manifest)):
        raise ValueError("Unbound physical name mapping")
    inventory, anchors, summary = inspect_corpus(args.root, protected=protected,
        maximum_archives=args.max_archives, source_manifest=source_manifest,
        bindings=bindings["episodes"] if bindings else None)
    if bindings and bindings["inventory_sha256"] != summary["inventory_sha256"]:
        raise ValueError("Name bindings belong to a different immutable archive inventory")
    rejected = bindings.get("rejected_episodes", {}) if bindings else {}
    for anchor in anchors:
        reason = rejected.get(canonical(anchor["source_episode"]))
        if reason:
            anchor["label_status"] = "quarantined_binding_mismatch"
            anchor["label_audit"]["binding_quarantine"] = reason
    summary["binding_quarantined_episodes"] = len(rejected)
    summary["binding_quarantined_anchors"] = sum("binding_quarantine" in a["label_audit"] for a in anchors)
    summary["protected_groups_supplied"] = args.protected_groups is not None
    summary["protected_groups_sha256"] = file_sha(args.protected_groups) if args.protected_groups else None
    summary["source_manifest_sha256"] = file_sha(args.source_manifest) if args.source_manifest else None
    summary["bindings_sha256"] = file_sha(args.bindings) if args.bindings else None
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("inventory.json", inventory), ("summary.json", summary),
                        ("review_selection.json", select_review(inventory))):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (args.output / "anchors.jsonl").open("x") as stream:
        for row in anchors:
            stream.write(canonical(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

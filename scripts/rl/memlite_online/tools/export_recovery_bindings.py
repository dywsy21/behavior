"""Read exact task metadata bindings on the simulator host, without loading it.

Output is privileged LABEL/AUDIT provenance, never deployment actor input.
The frozen run selects the precise full/partial template; its corresponding
instance state must exist and have scope keys compatible with that template.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recovery_corpus import canonical, digest, file_sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads(args.source_manifest.read_text())
    inventory = json.loads(args.inventory.read_text())
    contracts = {t["task"]: t for g in run["groups"] for t in g["tasks"]}
    templates, episodes, evidence = {}, {}, {}
    for item in inventory:
        ep = item["episode"]
        task = ep["task"]
        if ep["instance_id"] not in contracts[task]["train_instances"]:
            raise ValueError("Non-TRAIN instance")
        if task not in templates:
            mode = contracts[task]["template_mode"]
            if mode not in {"full_scene", "partial_rooms"}:
                raise ValueError("Unknown frozen template mode")
            ending = "_template.json" if mode == "full_scene" else "_template-partial_rooms.json"
            matches = list(args.task_root.glob("scenes/*/json/*_task_" + task + "_0_0" + ending))
            if len(matches) != 1:
                raise ValueError("No unique exact task template: " + task)
            path = matches[0]
            mapping = json.loads(path.read_text())["metadata"]["task"]["inst_to_name"]
            mapping = {key: name for key, name in mapping.items() if isinstance(name, str) and name}
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("Ambiguous inverse task binding: " + task)
            templates[task] = (path, mapping)
        path, mapping = templates[task]
        key = canonical([ep["run"], ep["episode_id"]])
        if key in episodes:
            continue
        stem = path.name.split("_task_" + task + "_0_0")[0] + "_task_" + task
        instance = path.parent / (stem + "_instances") / (stem + f"_0_{ep['instance_id']}_template-tro_state.json")
        state_keys = set(json.loads(instance.read_text()))
        if not state_keys or not state_keys <= set(mapping):
            raise ValueError("Instance scope differs from selected template: " + str(instance))
        episodes[key] = {name: entity for entity, name in mapping.items()}
        evidence[key] = dict(task=task, instance_id=ep["instance_id"],
                             template=str(path), template_sha256=file_sha(path),
                             instance_state=str(instance), instance_state_sha256=file_sha(instance),
                             physical_scope_keys=sorted(state_keys), actor_input_permitted=False)
    result = dict(schema="recovery_physical_bindings_v1", inventory_sha256=digest(inventory),
                  source_manifest_sha256=file_sha(args.source_manifest), episodes=episodes,
                  evidence=evidence, actor_input_permitted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(episodes=len(episodes), tasks=len(templates), output=str(args.output), sha256=file_sha(args.output))))


if __name__ == "__main__":
    main()

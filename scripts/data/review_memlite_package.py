"""Make readable, sample-ID-addressed review sheets; never approve a dataset."""
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def choose_rows(rows, stages):
    chosen = []
    groups = [
        [r for r in rows if r["memlite_branch"] == "high" and r["intent_status"] == "REPLAN"],
        [r for r in rows if r["memlite_branch"] == "low" and stages[r["step"]] == "brake"],
        [r for r in rows if r["memlite_branch"] == "low" and stages[r["step"]] == "align"],
        [r for r in rows if r["memlite_branch"] == "low" and stages[r["step"]] == "approach"],
        [r for r in rows if r["memlite_branch"] == "high" and "original manipulation remains pending" in r["memory_update"]],
    ]
    for i, group in enumerate(groups):
        if not group:
            # Some real resets are already within the local standoff radius:
            # their correct recovery is brake+align+settle, with no approach.
            # Inspect an additional real low-level frame, never invent a phase.
            used = {r["review_id"] for r in chosen}
            group = [r for r in rows if r["memlite_branch"] == "low" and r["review_id"] not in used]
            if not group:
                raise ValueError(f"Insufficient distinct real review rows for phase {i}")
        chosen.append(group[len(group)//2] if i in (2, 3) else group[0])
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tasks", type=int, nargs="+", choices=range(5), default=list(range(5)))
    args = ap.parse_args()
    data = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    all_reviews = []
    for task in args.tasks:
        for split in ("train", "eval"):
            candidates = sorted([r for r in data["trajectories"] if r["task_id"] == task and r["split"] == split],
                                key=lambda r: r["instance_id"])
            if len(candidates) < 2:
                raise ValueError(f"Need two {split} trajectories for task {task}")
            # Spread train examples, inspect both heldout source instances.
            trajectories = [candidates[0], candidates[-1]]
            for trajectory in trajectories:
                root = Path(trajectory["path"])
                m = json.loads((root / "manifest.json").read_text())
                with np.load(root / "trajectory.npz", allow_pickle=False) as f:
                    states, actions = f["states"], f["actions"]
                rows = sorted([r for r in data["samples"][split] if r["trajectory"] == str(root)], key=lambda r: r["step"])
                geometry = m["geometry_diagnostic_only"]
                chosen = choose_rows(rows, [r["stage"] for r in geometry])
                # Two/three rows per page keeps text and all three cameras
                # readable at native resolution without aspect distortion.
                for page, page_rows in enumerate((chosen[:3], chosen[3:])):
                    canvas = Image.new("RGB", (1344, len(page_rows)*540), "white")
                    draw = ImageDraw.Draw(canvas)
                    for n, row in enumerate(page_rows):
                        step = row["step"]
                        with np.load(root / f"obs_{step:05d}.npz", allow_pickle=False) as f:
                            for col, suffix in enumerate(("zed_link:Camera:0::rgb", "left_realsense_link:Camera:0::rgb", "right_realsense_link:Camera:0::rgb")):
                                key = next(k for k in f.files if k.endswith(suffix))
                                picture = Image.fromarray(f[key][..., :3])
                                picture.thumbnail((448, 288))
                                canvas.paste(picture, (col*448+(448-picture.width)//2, n*540))
                        g = geometry[step]
                        a = actions[step:step+16, :3]
                        lines = [f"{row['review_id']} / {split} / {g['stage']} | status={row['intent_status']} | observed yaw={states[step,2]:+.3f}",
                            f"expert next16 mean xy/yaw={np.round(a.mean(0),3).tolist()} | yaw range={a[:,2].min():+.3f}..{a[:,2].max():+.3f} | diagnostic heading={g['heading_error']:+.3f} distance={g['distance']:.3f}",
                            f"IN memory: {row['memory']}", f"IN previous intent: {row['previous_intent']}",
                            f"IN feedback: {row['execution_feedback']}",
                            f"{'TARGET' if row['memlite_branch'] == 'high' else 'IN'} intent: {row['intent']}",
                            f"{'TARGET' if row['memlite_branch'] == 'high' else 'NOT used by low'} memory update: {row['memory_update']}"]
                        rendered = [piece for line in lines for piece in textwrap.wrap(line, width=132)]
                        if len(rendered) > 12:
                            raise ValueError("Review text would be clipped")
                        draw.multiline_text((8, n*540+290), "\n".join(rendered), font=font, fill="black", spacing=2)
                        all_reviews.append({**row, "observed_base_velocity": states[step, :3].tolist(),
                            "action16_base": a.tolist(), "geometry_diagnostic_only": g,
                            "history_steps": list(range(step-80, step+1, 16)),
                            "sheet": f"task{task}_{split}_i{m['instance_id']}_p{page}.jpg"})
                    canvas.save(args.output_dir / f"task{task}_{split}_i{m['instance_id']}_p{page}.jpg", quality=95)
    (args.output_dir / "review_items.json").write_text(json.dumps({"manual_approval": False,
        "payload_sha256": data["review_payload_sha256"], "items": all_reviews}, indent=2) + "\n")
    print(json.dumps({"samples_to_inspect": len(all_reviews), "approval_granted": False}), flush=True)


if __name__ == "__main__":
    main()

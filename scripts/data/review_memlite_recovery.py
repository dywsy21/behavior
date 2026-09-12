"""Render factual observation/label audit sheets; never grants data approval."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = []
    for path in sorted(args.collection.glob("task*_instance*/manifest.json")):
        m = json.loads(path.read_text())
        if not m.get("accepted"):
            continue
        geometry = m["geometry_diagnostic_only"]
        with np.load(path.parent / "trajectory.npz", allow_pickle=False) as data:
            states, actions = data["states"], data["actions"]
        phases = [r["stage"] for r in geometry]
        candidates = [0, m["recovery_start"] - 16, m["recovery_start"],
                      m["recovery_start"] + 32, phases.index("approach"), phases.index("settle")]
        steps = list(dict.fromkeys(min((len(actions)-1)//16*16, ((s+15)//16)*16) for s in candidates))
        canvas = Image.new("RGB", (3*384, len(steps)*318), "white")
        draw = ImageDraw.Draw(canvas)
        record = {"trajectory": str(path.parent), "quality": m["quality"], "observations": []}
        for n, step in enumerate(steps):
            with np.load(path.parent / f"obs_{step:05d}.npz", allow_pickle=False) as data:
                for col, suffix in enumerate(("zed_link:Camera:0::rgb", "left_realsense_link:Camera:0::rgb", "right_realsense_link:Camera:0::rgb")):
                    key = next(k for k in data.files if k.endswith(suffix))
                    image = Image.fromarray(data[key][..., :3])
                    image.thumbnail((384, 276))
                    canvas.paste(image, (col*384 + (384-image.width)//2, n*318))
            row = geometry[step]
            line = f"t{m['task_id']} i{m['instance_id']} step={step} {row['stage']} | state yaw={states[step,2]:+.3f}"
            draw.text((5, n*318+278), line, fill="black")
            line = f"next16 yaw={np.mean(actions[step:step+16,2]):+.3f} | heading error={row['heading_error']:+.3f} | distance={row['distance']:.3f}"
            draw.text((5, n*318+296), line, fill="black")
            record["observations"].append({"step": step, "stage": row["stage"], "state": states[step].tolist(),
                "action16": actions[step:step+16].tolist(), "geometry_diagnostic_only": row})
        canvas.save(args.output_dir / f"task{m['task_id']}_instance{m['instance_id']:03d}.jpg", quality=94)
        audit.append(record)
    (args.output_dir / "audit.json").write_text(json.dumps({"manual_approval": False, "trajectories": audit}, indent=2) + "\n")


if __name__ == "__main__":
    main()

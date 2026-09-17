"""Four no-actuation sanity requests. Not robot policy performance or training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.client import request_action


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--uri",required=True)
    args=p.parse_args(); src=Path(args.input); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    images={n:np.asarray(Image.open(src/f"{n}.png").convert("RGB"))
            for n in ("head_rgb","left_wrist_rgb","right_wrist_rgb")}
    empty={n:images["left_wrist_rgb"] for n in images}
    cases=[("explicit_right_up",images,"Interface diagnostic, NOT a task rollout. Output exactly R UP FINE, regardless of the images.","R UP FINE"),
           ("explicit_base_left",images,"Interface diagnostic, NOT a task rollout. Output exactly BASE LEFT COARSE, regardless of the images.","BASE LEFT COARSE"),
           ("red_visible",images,"Visual diagnostic, NOT a task rollout. Look ONLY at the HEAD image. If a red object is visible, output R OPEN. Otherwise output L OPEN. Do not manipulate anything; just give the diagnostic code.","R OPEN"),
           ("no_red_floor_repeated_views",empty,"Visual diagnostic, NOT a task rollout. Look ONLY at the HEAD image. If a red object is visible, output R OPEN. Otherwise output L OPEN. Do not manipulate anything; just give the diagnostic code.","L OPEN")]
    rows=[]
    for name,views,text,expected in cases:
        result=request_action(args.uri,views,text,timeout=120)
        row={"case":name,"expected":expected,"response":result,"exact_match":result["action"]==expected,
             "acted":False,"synthetic_view_rebinding":name.startswith("no_red")}
        rows.append(row); print(json.dumps(row),flush=True)
    (out/"result.json").write_text(json.dumps({"kind":"input_sensitivity_not_rollout","cases":rows},indent=2))


if __name__=="__main__":
    main()

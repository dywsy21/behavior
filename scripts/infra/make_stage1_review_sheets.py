"""Arrange already-exported diagnostic frames for the owner's visual review.

This does not modify source RGB or training targets and does not approve data.
"""
import argparse
import json
from pathlib import Path
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads((args.qa / "rows.json").read_text())
    selection = json.loads((args.qa / "selection.json").read_text())
    by_locator = {(r["split"], r["candidate"]): i for i, r in enumerate(rows)}
    # A held-out frame in each task; every verb; long and boundary windows.
    chosen = {next(i for i,r in enumerate(rows) if r["task"] == task and r["split"] == "eval") for task in range(100)}
    chosen.update(by_locator[tuple(pair)] for pair in selection["skills"].values())
    chosen.update(i for i in sorted(range(len(rows)), key=lambda i: rows[i]["tokens"]["sequence_tokens"])[-16:])
    chosen.update(i for i in sorted(range(len(rows)), key=lambda i: rows[i]["valid_action_steps"])[:16])
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    chosen = sorted(chosen)
    for page, start in enumerate(range(0, len(chosen), 8)):
        indices = chosen[start:start+8]
        sheet = Image.new("RGB", (1440, 1500), "#cccccc")
        for slot, index in enumerate(indices):
            source = Image.open(args.qa / "images" / rows[index]["image"])
            sheet.paste(source.resize((720,375)), ((slot%2)*720, (slot//2)*375))
        name = f"page-{page:02d}.jpg"
        sheet.save(args.output / name, quality=95)
        records.append(dict(page=name, row_indices=indices,
                            identities=[{k:rows[i][k] for k in ("task","episode","frame","split","candidate","image")} for i in indices]))
    (args.output / "index.json").write_text(json.dumps(dict(status="AWAITING_HUMAN_REVIEW", rows=len(chosen), pages=records),indent=2)+"\n")
    print(json.dumps(dict(rows=len(chosen), pages=len(records), output=str(args.output))))


if __name__ == "__main__":
    main()

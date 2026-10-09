"""Render original, unenhanced candidate RGB for stratified primary review.

This is a contact sheet, not a label generator. The output cannot approve data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap
import zipfile

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recovery_corpus import CAMERAS, file_sha, select_review  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((args.audit / "inventory.json").read_text())
    positive_paths = set()
    with (args.audit / "anchors.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["label_audit"]["outcome"]["value"] is not None:
                positive_paths.add(row["actor_input"]["rgb"]["archive"])
    # Review ALL clips proposing a positive, plus >=60 task/event-stratified clips.
    selected = select_review(inventory, 60)
    paths = {r["path"] for r in selected}
    selected += [r for r in inventory if r["path"] in positive_paths - paths]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    reviews, page, page_paths = [], None, []
    for i, item in enumerate(selected):
        if i % 4 == 0:
            page = Image.new("RGB", (2040, 4 * 342), "white")
            draw = ImageDraw.Draw(page)
        path = args.root / item["path"]
        if file_sha(path) != item["sha256"]:
            raise ValueError("Archive changed before manual review")
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            rows = {r["control_step"]: r for r in
                    (json.loads(line) for line in archive.read("transitions.jsonl").splitlines())}
            anchors = sorted(int(t) for t in manifest["rgb_anchors"] if int(t) in rows)
            points = [anchors[0], anchors[len(anchors)//2], anchors[-1]]
            top = (i % 4) * 342
            events = ",".join(sorted({e["kind"] for e in item["events"]}))
            title = f"{i+1:02d} {item['episode']['task']} / instance {item['episode']['instance_id']} / {item['path']}"
            draw.text((4, top+2), title, font=font, fill="black")
            draw.text((4, top+21), "CANDIDATE ONLY | " + events, font=font, fill="black")
            frame_details = []
            for j, t in enumerate(points):
                context = rows[t]["context"]
                x = 4 + j * 680
                for c, camera in enumerate(CAMERAS):
                    name = f"rgb/{t:08d}/{camera}.jpg"
                    content = archive.read(name)
                    expected = manifest["rgb_anchors"][str(t)]["sha256"][camera]
                    if hashlib.sha256(content).hexdigest() != expected:
                        raise ValueError("Image changed before rendering")
                    im = Image.open(io.BytesIO(content)).convert("RGB")
                    if im.size != (224, 224):
                        raise ValueError("Do not resize original review images")
                    page.paste(im, (x+c*224, top+42))
                skills = json.loads(context["active_skills_semantic_json"])
                caption = f"s[{t}] " + "; ".join(f"{s['verb']}({s.get('arm','?')}) {s.get('target','?')}" for s in skills)
                for offset, line in enumerate(textwrap.wrap(caption, 84)[:3]):
                    draw.text((x, top+268+offset*18), line, font=font, fill="black")
                before = rows.get(t-1, {}).get("physical_audit", {})
                frame_details.append(dict(step=t, context=context,
                    pre_observation_grasp_states=before.get("grasp_states"),
                    image_sha256=manifest["rgb_anchors"][str(t)]["sha256"]))
            reviews.append(dict(ordinal=i+1, archive=item["path"], sha256=item["sha256"],
                episode=item["episode"], frames=frame_details, proposed_positive=item["path"] in positive_paths,
                review_status="pending", action_approval=False, outcome_approval=False))
        if i % 4 == 3 or i == len(selected)-1:
            target = args.output / f"sheet_{i//4+1:02d}.png"
            page.save(target)
            page_paths.append(dict(path=target.name, sha256=file_sha(target)))
    result = dict(schema="recovery_media_review_materials_v1", rows=reviews, sheets=page_paths,
                  inventory_sha256=file_sha(args.audit / "inventory.json"),
                  all_proposed_positive_archives_covered=positive_paths <= {r["archive"] for r in reviews},
                  training_ready=False)
    (args.output / "review-materials.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(clips=len(reviews), sheets=len(page_paths), original_rgb_panels=len(reviews)*9)))


if __name__ == "__main__":
    main()

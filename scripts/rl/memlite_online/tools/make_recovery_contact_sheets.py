"""Reproducible human-QA sheets of unmodified candidate RGB and recorded intent."""
import argparse
import io
import json
from pathlib import Path
import sys
import textwrap
import zipfile

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from validate_recovery import validate_archive  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    index = []
    cameras = ['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb']
    for rank_root in sorted(args.root.glob('gpu_*')):
        selected = sorted(rank_root.glob('*.zip'))[:2]
        if not selected:
            continue
        tiles, clips = [], []
        for path in selected:
            checked = validate_archive(path)
            clips.append(checked)
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read('manifest.json'))
                steps = sorted(map(int, manifest['rgb_anchors']))
                rows = {r['control_step']: r for r in
                        (json.loads(line) for line in archive.read('transitions.jsonl').splitlines())}
                for step in [steps[0], steps[-1]]:
                    row = rows.get(step) or next(r for r in rows.values() if r['rgb_anchor_control_step'] == step)
                    images = [Image.open(io.BytesIO(archive.read(f'rgb/{step:08d}/{camera}.jpg'))).convert('RGB')
                              for camera in cameras]
                    label = (f"{rank_root.name} / {manifest['episode']['task']} / instance "
                             f"{manifest['episode']['instance_id']} / frame {step} / policy {row['policy_update']}")
                    intent = row['context']['active_skills_text']
                    tiles.append((images, label, intent))
        width = max(sum(image.width for image in images) for images, _, _ in tiles)
        height = sum(max(image.height for image in images) + 90 for images, _, _ in tiles)
        sheet = Image.new('RGB', (width, height), 'white')
        draw, y = ImageDraw.Draw(sheet), 0
        for images, label, intent in tiles:
            draw.text((4, y+2), label, fill='black', font=font)
            draw.text((4, y+20), '\n'.join(textwrap.wrap('Recorded intent: '+intent, width=90)[:3]),
                      fill='black', font=font)
            x = 0
            for camera, picture in zip(cameras, images, strict=True):
                draw.text((x+4, y+71), camera, fill='black', font=font)
                sheet.paste(picture, (x, y+90))
                x += picture.width
            y += max(image.height for image in images)+90
        target = args.out/(rank_root.name+'.png')
        sheet.save(target)
        index.append(dict(sheet=str(target.resolve()), clips=clips,
                          image_panels=len(tiles)*3, manual_review='pending', bc_eligible=False))
    (args.out/'index.json').write_text(json.dumps(index, indent=2))
    print(json.dumps(dict(sheets=len(index), image_panels=sum(r['image_panels'] for r in index))))


if __name__ == '__main__':
    main()

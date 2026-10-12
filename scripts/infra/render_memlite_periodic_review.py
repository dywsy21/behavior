"""Original TRAIN RGB plus old/current command for periodic-history review.

Contact sheets are diagnostic renderings only. They do not alter raw videos,
training images, labels, admission, or model weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import textwrap
import time

from behavior_champion.dataset import NativeV3Dataset, CAMERAS
from behavior_champion.release import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        raise ValueError('CPU-only review and fresh output required')
    if sha256(args.candidates) != args.candidate_sha256:
        raise ValueError('Changed periodic-history candidate artifact')
    from PIL import Image, ImageDraw, ImageFont
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()]
    selected = [next(row for row in rows if row['identity']['task'] == task and row['identity']['kind'] == 'transition')
                for task in range(100)]
    # Preserve both no-prior and genuine same-command controls, not just the
    # newly enriched transition class. Selection never sees model predictions.
    for task in range(0, 100, 10):
        for kind in ('initial', 'continuation'):
            selected.append(next(row for row in rows if row['identity']['task'] == task and row['identity']['kind'] == kind))
    data = NativeV3Dataset(args.release, 'train')
    episodes = {int(ep['row']['episode_index']): ep for ep in data.episodes}
    args.output.mkdir(parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    evidence, pages = [], []; started = time.monotonic()
    for page, start in enumerate(range(0, len(selected), 4)):
        subset = selected[start:start+4]
        canvas = Image.new('RGB', (1344, 390 * len(subset)), 'white'); draw = ImageDraw.Draw(canvas)
        page_rows = []
        for slot, row in enumerate(subset):
            identity, target = row['identity'], row['projection']
            episode = episodes[identity['episode']]; source = episode['row']
            if episode['split'] != 'train' or int(source['task_index']) != identity['task']:
                raise ValueError('Foreign source in review candidate')
            frames = [identity['frame'] - 128, identity['frame']] if identity['kind'] != 'initial' else [identity['frame']]
            top = slot * 390
            lines = [f"#{start+slot} task={identity['task']} ep={identity['episode']} kind={identity['kind']} frames={frames}",
                     *textwrap.wrap('PREVIOUS ISSUED: ' + target['previous_intent'], 175),
                     *textwrap.wrap('CURRENT TARGET: ' + target['active_skills_text'], 175),
                     *textwrap.wrap('MEMORY: ' + target['memory'], 175)]
            for number, line in enumerate(lines[:10]):
                draw.text((4, top + number * 14), line, font=font, fill='black')
            hashes = {}
            for instant, frame in enumerate(frames):
                if frame < 0: raise ValueError('Invented pre-episode image')
                for camera, (name, sensor) in enumerate(CAMERAS.items()):
                    rgb = data._image(source, frame, sensor)
                    hashes[f'{frame}:{name}'] = hashlib.sha256(rgb.tobytes()).hexdigest()
                    canvas.paste(Image.fromarray(rgb).resize((224, 224)), (instant * 672 + camera * 224, top + 156))
            evidence.append(dict(identity=identity, projection=target, rgb_frames=frames, original_rgb_sha256=hashes,
                                 sheet_row=slot, original_candidate_index=identity['original_candidate_index']))
            page_rows.append(len(evidence)-1)
        name = f'page-{page:02d}.jpg'; canvas.save(args.output / name, quality=95, subsampling=0)
        pages.append(dict(path=name, sha256=sha256(args.output / name), rows=page_rows))
        print(json.dumps(dict(pages_completed=len(pages), cases_completed=len(evidence))), flush=True)
    result = dict(status='original_rgb_rendered_requires_owner_review', candidate_sha256=args.candidate_sha256,
        cases=len(selected), tasks=100, pages=pages, rows=evidence, seconds=time.monotonic()-started,
        human_review_complete=False, training_admitted=False, optimizer_updates=0,
        original_rgb_panels=sum(3 * len(row['rgb_frames']) for row in evidence))
    (args.output / 'review.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()

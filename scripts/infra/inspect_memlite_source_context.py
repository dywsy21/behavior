"""CPU original-RGB/annotation context for one disputed TRAIN candidate.

No relabeling, model prediction, optimization or implicit training admission.
"""
import argparse
import json
import os
from pathlib import Path
import textwrap

from behavior_champion.dataset import NativeV3Dataset, CAMERAS
from behavior_champion.release import AcceptedRelease, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        raise ValueError('CPU-only and fresh output required')
    selected = None
    for _, episode in AcceptedRelease(args.release).episodes('train'):
        if int(episode['row']['episode_index']) == args.episode: selected = episode
    if selected is None: raise ValueError('Source not in admitted original TRAIN')
    frames = [args.frame + delta for delta in (-768, -512, -256, -128, 0, 128, 256, 512, 768)]
    frames = [frame for frame in frames if 0 <= frame < int(selected['row']['length'])]
    data = NativeV3Dataset(args.release, 'train')
    from PIL import Image, ImageDraw, ImageFont
    args.output.mkdir(parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    records, pages = [], []
    for page, offset in enumerate(range(0, len(frames), 3)):
        canvas = Image.new('RGB', (1536, 570*len(frames[offset:offset+3])), 'white')
        draw = ImageDraw.Draw(canvas)
        for slot, frame in enumerate(frames[offset:offset+3]):
            active = [segment for segment in selected['segments'] if segment['start'] <= frame < segment['end']]
            if len(active) > 1: raise ValueError('Overlapping compiled source segments')
            title = f'ep={args.episode} task={selected["row"]["task_index"]} local_frame={frame}'
            label = active[0]['text'] if active else 'UNANNOTATED: not guessed'
            for line, text in enumerate([title, *textwrap.wrap(label, 165)]):
                draw.text((4, slot*570+line*19), text, font=font, fill='black')
            for camera, sensor in enumerate(CAMERAS.values()):
                rgb = data._image(selected['row'], frame, sensor)
                rendered = Image.fromarray(rgb); rendered.thumbnail((512, 440))
                canvas.paste(rendered, (camera*512, slot*570+125))
            records.append(dict(frame=frame, source_segment=active))
        path = args.output / f'context-{page:02d}.jpg'; canvas.save(path, quality=95, subsampling=0)
        pages.append(dict(path=path.name, sha256=sha256(path)))
    result = dict(status='original_context_for_review_no_label_change', episode=args.episode,
                  requested_frame=args.frame, original_row=selected['row'], frames=records, pages=pages,
                  training_admitted=False, optimizer_updates=0)
    (args.output / 'context.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], frames=len(records), pages=len(pages))), flush=True)


if __name__ == '__main__': main()

"""Validate/render/export the separately corrected H82 image-only labels.

Never edits frozen teacher predictions, H76 gold, source images, or a trainer.
Uncertain rows remain in the audit but cannot enter the definite-label export.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from visual_grounding import canonical_target, parse_answer

REPO = Path(__file__).resolve().parents[2]
REVIEW = REPO / 'configs/vlm_sft/h82_parent_corrected_grounding_v1.json'
PINS = {
    'manifest_sha256': '916ecc58b0817983f322dde5ae5db7f835b68b27d6e41902e53f1ec17b8b19a7',
    'predictions_sha256': 'a231bbab2c4fa66f7f8fda304f2ef73c73ab0eff8c6ac1f9fb3bfc6fc7fb5cf1',
    'frozen_h76_review_sha256': '3159ed61cb56778343e360e282a9fbb68b64e5dd7d770d0346397fa1380dd136',
    'input_protocol_sha256': '4b0d80a200a0553839e4d6fa00df1e5978d13aa40fa22d858029397e574309d7',
    'h83_parent_review_sha256': 'a4c3628141056a2b2e31a2e8b6a36297c442a32a87d0a0a3945b0d7972a82010',
}
CLASSES = {'P': 'present', 'N': 'absent', 'U': 'uncertain'}
VIEWS = ('head', 'left_wrist', 'right_wrist')
ROW_KEYS = {'id', 'task', 'instance', 'episode', 'frame', 'view', 'query', 'split', 'image',
            'png_sha256', 'pixels_sha256', 'resolution', 'frozen_visibility', 'source_teacher_target',
            'source_review_note', 'boxes_px', 'target', 'review_note', 'raw_native_reviewed',
            'overview_reviewed', 'training_eligible', 'quarantine_reason'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def normalized_boxes(boxes, resolution):
    w, h = resolution
    if resolution not in ([720, 720], [480, 480]):
        raise ValueError('Native camera resolution required')
    out = []
    for box in boxes:
        if (type(box) is not list or len(box) != 4 or any(type(x) is not int for x in box)
                or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h):
            raise ValueError('Invalid native pixel box')
        x1, y1, x2, y2 = box
        out.append([x1 * 1000 // w, y1 * 1000 // h,
                    (x2 * 1000 + w - 1) // w, (y2 * 1000 + h - 1) // h])
    return out


def validate_records(review, *, allow_draft=False):
    if (review['schema'] != 'h82-parent-corrected-grounding-v1'
            or review['protocol'] != 'visual-grounding-v1' or review['source_pins'] != PINS
            or review['label_origin'] != 'parent_manual_visual_correction'):
        raise ValueError('Unexpected review identity')
    if review['status'] not in ('DRAFT_PENDING_FINAL_OVERLAY_REVIEW', 'FINAL_REVIEW_PASSED'):
        raise ValueError('Unknown review status')
    rows = review['rows']
    if len(rows) != 81 or len({r['id'] for r in rows}) != 81:
        raise ValueError('Exactly 81 unique primary images required')
    if len({(r['task'], r['instance']) for r in rows}) != 9:
        raise ValueError('Exactly nine existing TRAIN groups required')
    for row in rows:
        if set(row) != ROW_KEYS or row['split'] != 'visual_train' or row['view'] not in VIEWS:
            raise ValueError('Exact TRAIN image record required')
        target = parse_answer(canonical_target(row['target']))
        if target['visibility'] != CLASSES[row['frozen_visibility']]:
            raise ValueError('Frozen visibility changed; a separate erratum is required')
        if normalized_boxes(row['boxes_px'], row['resolution']) != target['boxes']:
            raise ValueError('Normalized/pixel boxes disagree')
        parse_answer(canonical_target(row['source_teacher_target']))
        eligible = target['visibility'] != 'uncertain'
        if row['training_eligible'] is not eligible:
            raise ValueError('Uncertain rows cannot be exported for training')
        if row['quarantine_reason'] != (None if eligible else 'identity_or_food_qualifier_unresolved_from_current_image'):
            raise ValueError('Missing uncertainty quarantine')
        if (row['overview_reviewed'] is not True or not row['review_note'].strip()
                or type(row['raw_native_reviewed']) is not bool
                or (target['visibility'] == 'present' and row['raw_native_reviewed'] is not True)):
            raise ValueError('Incomplete manual review provenance')
        if row['image'] != 'images/' + row['id'] + '.png':
            raise ValueError('Only the exact original PNG may be exported')
    counts = Counter(r['target']['visibility'] for r in rows)
    if counts != {'present': 32, 'absent': 41, 'uncertain': 8} or sum(len(r['target']['boxes']) for r in rows) != 33:
        raise ValueError('Corrected cohort coverage changed')
    if sum(r['target']['visibility'] != r['source_teacher_target']['visibility'] for r in rows) != 12:
        raise ValueError('Historical visibility correction count changed')
    if not allow_draft:
        proof = review['final_overlay_review']
        if review['status'] != 'FINAL_REVIEW_PASSED' or not proof or proof['records_sha256'] != records_sha(rows):
            raise ValueError('Final overlay review missing or stale')
        ids = [i for p in proof['pages'] for i in p['ids']]
        if (len(proof['pages']) != 9 or len(ids) != 81 or set(ids) != {r['id'] for r in rows}
                or proof['all_pages_inspected_by_parent'] is not True
                or proof['known_issues_remaining'] != 0):
            raise ValueError('Incomplete final image/label review')
        for i, page in enumerate(proof['pages']):
            if (set(page) != {'path', 'ids', 'sha256'} or Path(page['path']).name != f'corrected_{i:02d}.jpg'
                    or len(page['sha256']) != 64 or any(c not in '0123456789abcdef' for c in page['sha256'])):
                raise ValueError('Pinned final review page required')
    return rows


def verify_final_sheets(review, overlay_root=None):
    proof = review['final_overlay_review']
    root = Path(overlay_root) if overlay_root is not None else REPO / Path(proof['pages'][0]['path']).parent
    receipt = json.loads((root / 'sheets.json').read_text())
    if (receipt['purpose'] != 'human_only_final_label_review' or receipt['training_eligible'] is not False
            or receipt['records_sha256'] != records_sha(review['rows']) or len(receipt['pages']) != 9):
        raise ValueError('Final render receipt is stale')
    for approved, rendered in zip(proof['pages'], receipt['pages']):
        if (approved['ids'] != rendered['ids'] or approved['sha256'] != rendered['sha256']
                or Path(approved['path']).name != Path(rendered['path']).name):
            raise ValueError('Rendered page differs from parent approval')
        path = root / Path(approved['path']).name
        if path.is_symlink() or sha(path) != approved['sha256']:
            raise ValueError('Final overlay pixels changed')


def load_review(review_path, bundle, predictions, *, allow_draft=False, overlay_root=None):
    paths = {'manifest_sha256': bundle / 'review_manifest.json', 'predictions_sha256': predictions,
             'frozen_h76_review_sha256': REPO / 'configs/vlm_sft/h76_parent_visual_review_v1.json',
             'input_protocol_sha256': Path(__file__).with_name('visual_grounding.py'),
             'h83_parent_review_sha256': REPO / 'configs/vlm_sft/h83_parent_dual_teacher_review_v1.json'}
    if {key: sha(path) for key, path in paths.items()} != PINS:
        raise ValueError('Frozen source bytes changed')
    review = json.loads(review_path.read_text())
    rows = validate_records(review, allow_draft=allow_draft)
    if not allow_draft:
        verify_final_sheets(review, overlay_root)
    manifest = json.loads(paths['manifest_sha256'].read_text())
    gold = json.loads(paths['frozen_h76_review_sha256'].read_text())
    gold = {r['id']: r for r in gold['rows']}
    original = [json.loads(line) for line in predictions.read_text().splitlines()]
    primary = [r for r in original if r['phase'] == 'primary']
    if len(primary) != 81 or len(original) != 97 or len({r['id'] for r in primary}) != 81:
        raise ValueError('Primary/repeat source changed')
    primary = {r['id']: r for r in primary}
    source = {state['id'] + '_' + view: (state, view, j)
              for state in manifest['rows'] if state['candidate_split'] == 'visual_train'
              for j, view in enumerate(VIEWS)}
    if set(source) != {r['id'] for r in rows} or set(source) != set(primary):
        raise ValueError('Missing primary or included holdout/repeat')
    protected = {(r['task'], r['instance']) for r in json.loads(paths['h83_parent_review_sha256'].read_text())['rows']}
    for row in rows:
        state, view, j = source[row['id']]
        if (row['task'], row['instance']) in protected:
            raise ValueError('Protected H83 calibration source')
        if any(row[key] != state[key] for key in ('task', 'instance', 'episode', 'frame', 'query')) or row['view'] != view:
            raise ValueError('Source identity mismatch')
        label = gold[state['id']]['labels'][j]
        teacher = primary[row['id']]
        if (row['frozen_visibility'] != label or teacher['expected_visibility'] != label
                or row['source_teacher_target'] != teacher['parsed'] or row['query'] != teacher['query']
                or row['png_sha256'] != state['image_receipts'][view]['png_sha256']
                or row['png_sha256'] != teacher['png_sha256'] or row['image'] != state['images'][view]):
            raise ValueError('Gold/teacher/image provenance mismatch')
        path = bundle / row['image']
        if path.is_symlink() or path.parent.is_symlink() or sha(path) != row['png_sha256']:
            raise ValueError('Original PNG changed or aliased')
        with Image.open(path) as im:
            im.load()
            if (im.mode != 'RGB' or list(im.size) != row['resolution']
                    or hashlib.sha256(im.tobytes()).hexdigest() != row['pixels_sha256']):
                raise ValueError('Original pixels changed')
    return review


def render(review, bundle, output):
    rows = validate_records(review, allow_draft=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'native_positive').mkdir()
    pages = []
    for offset in range(0, len(rows), 9):
        group = rows[offset:offset + 9]
        canvas = Image.new('RGB', (1080, 1260), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(group):
            with Image.open(bundle / row['image']) as im:
                native = im.convert('RGB')
            painter = ImageDraw.Draw(native)
            w, h = native.size
            for x1, y1, x2, y2 in row['target']['boxes']:
                painter.rectangle([x1*w/1000, y1*h/1000, min(w-1,x2*w/1000), min(h-1,y2*h/1000)],
                                  outline=(0, 210, 50), width=2)
            if row['target']['visibility'] == 'present':
                native.save(output / 'native_positive' / (row['id'] + '.png'))
            x, y = (i % 3) * 360, (i // 3) * 420
            canvas.paste(native.resize((360, 360), Image.Resampling.LANCZOS), (x, y + 60))
            for line, label in enumerate((row['id'], row['query'], 'MANUAL: ' + row['target']['visibility']
                                         + ' | OLD: ' + row['source_teacher_target']['visibility'])):
                draw.text((x + 2, y + 2 + 18*line), label, fill=(0, 0, 0))
        path = output / f'corrected_{offset//9:02d}.jpg'
        canvas.save(path, quality=95)
        pages.append({'path': str(path), 'ids': [r['id'] for r in group], 'sha256': sha(path)})
    receipt = {'purpose': 'human_only_final_label_review', 'training_eligible': False,
               'records_sha256': records_sha(rows), 'pages': pages}
    (output / 'sheets.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def export(review_path, bundle, predictions, output, *, overlay_root=None):
    # The public exporter cannot bypass the original-source and image audit.
    review = load_review(review_path, bundle, predictions, overlay_root=overlay_root)
    review_bytes = review_path.read_bytes()
    if json.loads(review_bytes) != review:
        raise ValueError('Export review bytes/record mismatch')
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    rows = validate_records(review)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('Never write an export into the frozen source bundle')
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    with (output / 'train.jsonl').open('x') as train, (output / 'quarantine.jsonl').open('x') as quarantine:
        for row in rows:
            value = {k: row[k] for k in ('id', 'task', 'instance', 'episode', 'frame', 'view', 'query',
                                         'split', 'png_sha256', 'pixels_sha256', 'training_eligible')}
            value.update(path=str((bundle / row['image']).resolve()), protocol='visual-grounding-v1',
                         target=canonical_target(row['target']), label_origin=review['label_origin'],
                         label_review_sha256=review_sha, action_training_eligible=False,
                         independent_eval_eligible=False)
            target = train if row['training_eligible'] else quarantine
            target.write(json.dumps(value, separators=(',', ':')) + '\n')
            counts['train' if row['training_eligible'] else 'quarantine'] += 1
    # Only seal after a second full audit. A concurrent change leaves unsealed
    # diagnostics, not a silently mixed release. Consumers require release.json.
    if (load_review(review_path, bundle, predictions, overlay_root=overlay_root) != review
            or sha(review_path) != review_sha):
        raise ValueError('Sources/review changed while exporting; release not sealed')
    result = {'status': 'HUMAN_CORRECTED_IMAGE_ONLY_RELEASE', 'counts': dict(counts),
              'review_sha256': review_sha, 'records_sha256': records_sha(rows),
              'files': {name: sha(output / name) for name in ('train.jsonl', 'quarantine.jsonl')},
              'new_source_images': 0, 'new_training_updates': 0, 'h80_bulk_release': False,
              'historical_teacher_correct': 69, 'historical_teacher_total': 81}
    (output / 'release.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('render', 'export', 'validate'), required=True)
    parser.add_argument('--review', type=Path, default=REVIEW)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overlay-root', type=Path, help='Exact regenerated or archived parent-reviewed sheets')
    args = parser.parse_args()
    reviewed = None if args.mode == 'export' else load_review(
        args.review, args.bundle, args.predictions, allow_draft=args.mode == 'render', overlay_root=args.overlay_root)
    if args.mode in ('render', 'export') and args.output is None:
        parser.error('--output is required for render/export')
    if args.mode == 'render':
        result = render(reviewed, args.bundle, args.output)
    elif args.mode == 'export':
        result = export(args.review, args.bundle, args.predictions, args.output, overlay_root=args.overlay_root)
    else:
        result = {'status': 'VALIDATED', 'records_sha256': records_sha(reviewed['rows']), 'images': 81}
    print(json.dumps(result, indent=2))

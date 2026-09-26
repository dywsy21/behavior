"""Shared CPU encoding for the opt-in composite candidate, never legacy SFT.

The response is appended to the identical inference prefix. No silent context
truncation, source identities or supervised prompt tokens. This is a processor
adapter, not an image provenance verifier: the loader must bind CURRENT pixels
to a pinned row/receipt using trajectory_images before calling this module.
"""
from __future__ import annotations

from PIL import Image

import native_trajectory_codec as codec
from modeling import collate, eos_id, supervised_loss

VIEWS = ('head', 'left_wrist', 'right_wrist')
IMAGE_SIZES = {'head': 448, 'left_wrist': 320, 'right_wrist': 320}
MAX_CONTEXT = 4096
MAX_RESPONSE = 1536


def messages(actor, images):
    codec.validate_actor(actor)
    if type(images) is not dict or set(images) != set(VIEWS):
        raise ValueError('Exactly three current onboard images required')
    content = []
    for view in VIEWS:
        img = images[view]
        if not isinstance(img, Image.Image) or img.width < 1 or img.height < 1 or img.width != img.height:
            raise ValueError('Square onboard RGB image required')
        size = IMAGE_SIZES[view]
        rgb = img.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        content.extend([{'type': 'text', 'text': view.upper()}, {'type': 'image', 'image': rgb}])
    content.append({'type': 'text', 'text': codec.prompt(actor)})
    return [{'role': 'system', 'content': codec.SYSTEM}, {'role': 'user', 'content': content}]


def check_visual_tensor(prefix, processor):
    import torch
    if any(k not in prefix for k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')):
        raise ValueError('Actual multimodal tensors required, not a text-only prefix')
    if prefix['input_ids'].ndim != 2 or prefix['input_ids'].shape[0] != 1:
        raise ValueError('Single sample prefix required')
    if prefix['attention_mask'].shape != prefix['input_ids'].shape or not torch.all(prefix['attention_mask'] == 1):
        raise ValueError('Unpadded, untruncated single-sample prefix required')
    expected = torch.tensor([[1, 28, 28], [1, 20, 20], [1, 20, 20]], device=prefix['image_grid_thw'].device)
    grid = prefix['image_grid_thw']
    if grid.shape != (3, 3) or not torch.equal(grid, expected):
        raise ValueError('Registered Qwen3.5-2B three-view image grid changed')
    pixels = prefix['pixel_values']
    if pixels.ndim != 2 or tuple(pixels.shape) != (1584, 1536) or not torch.isfinite(pixels).all():
        raise ValueError('Missing, nonfinite, or wrong-shape three-view pixels')
    token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if token is None or processor.tokenizer.convert_ids_to_tokens(token) != '<|image_pad|>':
        raise ValueError('Verified native image placeholder required')
    if int((prefix['input_ids'] == token).sum()) != 396:
        raise ValueError('Exactly 396 merged visual tokens expected')


def encode(processor, actor, images, *, target=None):
    import torch
    prefix = processor.apply_chat_template(messages(actor, images), tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False)
    check_visual_tensor(prefix, processor)
    old = prefix['input_ids'].shape[1]
    if old >= MAX_CONTEXT: raise ValueError('Prefix cannot fit a response; no truncation allowed')
    if target is None: return prefix
    codec.parse(target)
    response = processor.tokenizer.encode(target, add_special_tokens=False)
    if processor.tokenizer.decode(response, skip_special_tokens=False) != target:
        raise ValueError('Tokenizer cannot preserve exact target JSON')
    response.append(eos_id(processor))
    if not 2 <= len(response) <= MAX_RESPONSE or old + len(response) > MAX_CONTEXT:
        raise ValueError('Full response exceeds frozen sequence budget; never truncate')
    result = {k: v.clone() for k, v in prefix.items()}
    result['input_ids'] = torch.cat([prefix['input_ids'], torch.tensor([response], dtype=torch.long)], dim=1)
    for key in ('attention_mask', 'token_type_ids', 'mm_token_type_ids'):
        if key in result:
            value = 1 if key == 'attention_mask' else 0
            result[key] = torch.cat([result[key], torch.full((1, len(response)), value, dtype=result[key].dtype)], dim=1)
    labels = torch.full_like(result['input_ids'], -100)
    labels[:, old:] = result['input_ids'][:, old:]
    result['labels'] = labels
    if not torch.equal(result['input_ids'][:, :old], prefix['input_ids']):
        raise RuntimeError('Training and inference prefixes diverged')
    return result

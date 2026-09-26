"""Pixel-independent prefix serialization for the fixed three-image protocol.

Must match the real processor's input_ids exactly before bulk export is allowed.
This computes text/visual-placeholder tokens, never fabricates visual tensors.
"""
import hashlib

import native_trajectory_codec as codec
from prepare_full_annotation import packed

VISUAL_TOKENS = (196, 100, 100)
VIEWS = ('head', 'left_wrist', 'right_wrist')


def prefix_ids(tokenizer, template, actor):
    codec.validate_actor(actor)
    content = []
    for view in VIEWS:
        content.extend([{'type': 'text', 'text': view.upper()}, {'type': 'image'}])
    content.append({'type': 'text', 'text': codec.prompt(actor)})
    messages = [{'role': 'system', 'content': codec.SYSTEM}, {'role': 'user', 'content': content}]
    text = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    pieces = text.split('<|image_pad|>')
    if len(pieces) != 4: raise ValueError('Exactly three original image placeholders required')
    expanded = pieces[0] + ''.join('<|image_pad|>' * count + piece
                                   for count, piece in zip(VISUAL_TOKENS, pieces[1:]))
    ids = tokenizer.encode(expanded, add_special_tokens=False)
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if ids.count(image_id) != sum(VISUAL_TOKENS): raise ValueError('Image expansion changed')
    return ids


def ids_digest(ids):
    # Identical shape/serialization to audit_trajectory_encoding's [1, sequence].
    return hashlib.sha256(packed([ids])).hexdigest()

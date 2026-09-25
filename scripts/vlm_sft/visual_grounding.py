"""Image-only visibility and 2D grounding contract, shared by SFT and inference.

This is not a contact, completion, or action label. Teacher outputs are proposals
until a separate manual data release; syntactic validity is not correctness.
"""
import hashlib
import json
from PIL import Image

from modeling import eos_id

VERSION='visual-grounding-v1'
QUERIES=('the radio','a soda can','the wastebasket','a pumpkin','a candle','a cabinet','the cauldron',
         'a table','the plate holding food','a refrigerator','a bowl','a sink','a hinged jar','a cooked bratwurst')
SYSTEM='''Inspect only the supplied CURRENT_RAW image for the queried target. Return exactly one JSON object with two keys: "visibility" and "boxes".
"visibility" is "present", "absent", or "uncertain". "boxes" is a list of bounding boxes [x1,y1,x2,y2] using integer coordinates normalized to 0..1000 along the displayed image width and height. For example {"visibility":"present","boxes":[[100,200,350,450]]}.
present: at least one visually identifiable target is visible, including identifiable partial views. Draw tight boxes around visible target extents, clipped to the image; do not guess hidden extents. Include every clearly identifiable matching object (up to 8), not robot parts or neighboring objects. A present answer must have at least one box.
absent: no identifiable target pixels are visible; this does NOT mean the object does not exist elsewhere or behind an occluder. Use an empty boxes list.
uncertain: a candidate fragment or object is visible but its identity or a query qualifier cannot be confirmed from this image. Use an empty boxes list. If a plate is viewed from its back, do not assume food is on it.
Do not infer holding, contact, motion, task progress, success, or the next action. Return JSON only, with no markdown or explanation.'''


def parse_answer(text):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    value=json.loads(text,object_pairs_hook=unique)
    if type(value) is not dict or set(value)!={'visibility','boxes'}:
        raise ValueError('Exact visibility/boxes schema required')
    state=value['visibility'];boxes=value['boxes']
    if state not in ('present','absent','uncertain') or type(boxes) is not list or len(boxes)>8:
        raise ValueError('Invalid visibility or boxes')
    if (state=='present') != bool(boxes): raise ValueError('Only identifiable present targets have boxes')
    for box in boxes:
        if (type(box) is not list or len(box)!=4 or any(type(v) is not int or not 0<=v<=1000 for v in box) or
                not box[0]<box[2] or not box[1]<box[3]):
            raise ValueError('Strict integer normalized visible box required')
    if len({tuple(b) for b in boxes})!=len(boxes): raise ValueError('Duplicate boxes')
    return value


def decode_response(processor,generated):
    """Keep raw token evidence; never hide an invalid special token in JSON."""
    tokenizer=processor.tokenizer;eos=eos_id(processor);pad=tokenizer.pad_token_id
    text='';parsed=None;error=None;has_eos=eos in generated
    tokens=generated[:generated.index(eos)] if has_eos else generated
    try:
        if (type(generated) is not list or not 0<len(generated)<=192 or
                any(type(i) is not int or not 0<=i<len(tokenizer) for i in generated)):
            raise ValueError('Invalid raw generated token IDs')
        text=tokenizer.decode(tokens,skip_special_tokens=False).strip()
        if not has_eos:raise ValueError('Generation truncated without EOS')
        if any(i in tokenizer.all_special_ids for i in tokens):raise ValueError('Special token before terminal EOS')
        if any(i!=pad for i in generated[len(tokens)+1:]):raise ValueError('Non-padding tokens after terminal EOS')
        parsed=parse_answer(text)
    except (ValueError,TypeError) as caught:error=repr(caught)
    return {'text':text,'parsed':parsed,'format_valid':error is None,'error':error,'has_eos':has_eos,
            'output_tokens':len(tokens)+(1 if has_eos else 0),'generated_token_ids':generated}


def canonical_target(value):
    text=json.dumps(value,separators=(',',':'),allow_nan=False)
    parse_answer(text)
    return text


def messages(query,image,*,blind=False):
    if query not in QUERIES: raise ValueError('Unregistered target query')
    img=image.convert('RGB').copy()
    if img.size not in ((720,720),(480,480)): raise ValueError('Original onboard image required')
    img.thumbnail((640,640))
    if blind: img=Image.new('RGB',img.size,(128,128,128))
    msg=[{'role':'system','content':SYSTEM},{'role':'user','content':[
        {'type':'text','text':'CURRENT_RAW'},{'type':'image','image':img},{'type':'text','text':'Query: '+query}]}]
    return msg,{'protocol':VERSION,'size':list(img.size),'blind':blind,
                'resized_pixels_sha256':hashlib.sha256(img.tobytes()).hexdigest()}


def encode(processor,row,image,*,supervised=False,blind=False):
    import torch
    msg,receipt=messages(row['query'],image,blind=blind)
    prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',enable_thinking=False)
    old=prefix['input_ids'].shape[1]
    if old>1800: raise ValueError('Image-only grounding context budget exceeded')
    receipt.update(input_tokens=old,input_ids_sha256=hashlib.sha256(prefix['input_ids'].numpy().tobytes()).hexdigest())
    if not supervised: return prefix,receipt
    if blind: raise ValueError('Gray ablations are evaluation-only')
    target=canonical_target(parse_answer(row['target']))
    response=processor.tokenizer.encode(target,add_special_tokens=False)+[eos_id(processor)]
    if not 2<=len(response)<=192: raise ValueError('Grounding response token budget exceeded')
    result={k:v.clone() for k,v in prefix.items()}
    result['input_ids']=torch.cat([prefix['input_ids'],torch.tensor([response],dtype=torch.long)],dim=1)
    for key in ('attention_mask','token_type_ids','mm_token_type_ids'):
        if key in result:
            fill=1 if key=='attention_mask' else 0
            result[key]=torch.cat([result[key],torch.full((1,len(response)),fill,dtype=result[key].dtype)],dim=1)
    labels=torch.full_like(result['input_ids'],-100);labels[:,old:]=result['input_ids'][:,old:]
    result['labels']=labels
    return result,receipt

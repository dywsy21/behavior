"""Plain local HTTP transport with no hidden inference retries."""
import base64
from io import BytesIO
import json
import time
from urllib.request import Request, urlopen

from PIL import Image

from .prompts import SYSTEM, user_text


def request_action(uri, images, text, timeout=60):
    encoded=[]
    for name in ("head_rgb","left_wrist_rgb","right_wrist_rgb"):
        image=images[name]
        if image.shape[0] == 3:
            image=image.transpose(1,2,0)
        buf=BytesIO(); Image.fromarray(image).save(buf,format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    data={"system":SYSTEM,"text":text,"images":encoded}
    start=time.perf_counter()
    req=Request(uri,data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=timeout) as reply:
        result=json.load(reply)
    result["client_roundtrip_s"]=time.perf_counter()-start
    return result

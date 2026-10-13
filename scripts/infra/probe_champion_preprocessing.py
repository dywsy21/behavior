import hashlib,json,pathlib,sys,time,importlib.util
import numpy as np
import PIL
started=time.monotonic()
native=pathlib.Path(sys.argv[1]); implementation=pathlib.Path(sys.argv[2])
expected='fe19d18c56b4999b2df766a38236b4cc42751cfeef0bc11ce7270f26a114e0ad'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8<<20): h.update(b)
 return h.hexdigest()
assert sha(native)==expected
spec=importlib.util.spec_from_file_location('pinned_image_tools',implementation);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=np.load(native,allow_pickle=False);rows=[];total=hashlib.sha256()
assert source['task_id'].tolist()==list(range(100))
for camera in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'):
 pixels=source[camera]; assert len(pixels)==100 and pixels.dtype==np.uint8
 for task,image in enumerate(pixels):
  result=module.resize_with_pad(image,224,224);assert result.shape==(224,224,3) and result.dtype==np.uint8
  total.update(camera.encode()+str(task).encode()+result.tobytes())
  rows.append(dict(task=task,camera=camera,input_shape=list(image.shape),input_sha256=hashlib.sha256(image.tobytes()).hexdigest(),output_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
print(json.dumps(dict(schema='champion_cpu_preprocess_comparison_v1',native_sha256=expected,source=str(native),implementation=str(implementation),implementation_sha256=sha(implementation),pillow=PIL.__version__,numpy=np.__version__,images=len(rows),results=rows,total_output_sha256=total.hexdigest(),seconds=time.monotonic()-started,gpu_allocations=0,public_samples=0,model_calls=0)))

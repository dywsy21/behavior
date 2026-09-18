"""Frame/clock/split audit and human-review panels for frozen H-09 data."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.kinematics import RobotModel
from common import CAMERAS, POSITIONS, QUATERNIONS, sha, write_json
from prepare import ROOT, LABELS, RELEASE, reserve


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--panels",action="store_true")
    args=p.parse_args();reserve(args.data)
    groups=json.loads((args.data/"groups.json").read_text())
    model=RobotModel(json.loads(args.calibration.read_text()))
    audit=[];counts=Counter()
    for ep,g in sorted(groups.items(),key=lambda x:int(x[0])):
        e=g["episode"];ep=int(ep)
        path=ROOT/f'data/chunk-{e["data/chunk_index"]:03d}/file-{e["data/file_index"]:03d}.parquet'
        cols=["frame_index","timestamp","observation.state"]+["observation.robot2cam_pose."+v for v in CAMERAS.values()]
        t=pq.read_table(path,columns=cols,filters=[("episode_index","=",ep)]).sort_by("frame_index")
        frames=t["frame_index"].to_numpy();timestamps=t["timestamp"].to_numpy()
        if not np.allclose(timestamps-timestamps[0],frames/30,atol=.0002):
            raise RuntimeError("Expert clock is not aligned at 30 Hz")
        for f in np.linspace(0,len(t)-1,3,dtype=int):
            row=t.slice(int(f),1).to_pylist()[0];s=np.asarray(row["observation.state"])
            q=np.r_[s[53:57],s[3:10],s[28:35]]
            for arm in POSITIONS:
                T=model.forward(q,arm)
                error=float(np.linalg.norm(T[:3,3]-s[POSITIONS[arm]]))
                angle=float(np.linalg.norm((Rotation.from_matrix(T[:3,:3])*Rotation.from_quat(s[QUATERNIONS[arm]]).inv()).as_rotvec()))
                audit.append({"episode":ep,"frame":int(f),"link":arm,"position_m":error,"angle_rad":angle})
            for view,camera in CAMERAS.items():
                pose=np.asarray(row["observation.robot2cam_pose."+camera]);T=model.forward(q,"camera_"+view)
                error=float(np.linalg.norm(T[:3,3]-pose[:3]))
                angle=float(np.linalg.norm((Rotation.from_matrix(T[:3,:3])*Rotation.from_quat(pose[3:]).inv()).as_rotvec()))
                audit.append({"episode":ep,"frame":int(f),"link":"camera_"+view,"position_m":error,"angle_rad":angle})
        counts[g["split"]]+=1
    maxpos=max(r["position_m"] for r in audit);maxang=max(r["angle_rad"] for r in audit)
    passed=maxpos<.003 and maxang<.02
    receipt={"passed":passed,"clock_hz":30,"fixed_window_s":16/30,"calibration_sha256":sha(args.calibration),
        "calibration_path":str(args.calibration),"model_scene_truth":model.spec["metadata"].get("scene_truth"),
        "source_labels_sha256":sha(LABELS),"source_labels_expected_sha256":json.loads((RELEASE/"composite_release_manifest.json").read_text())["base_r2"]["labels_sha256"],
        "split_episode_counts":dict(counts),"frame_comparisons":audit,"max_position_m":maxpos,"max_angle_rad":maxang,
        "meaning":"Expert EEF/camera state is independently reproduced by robot-only base-frame FK; no world/object pose used.",
        "action_scale_limitation":"Supervision predicts direction only. Expert 16-frame displacement is not claimed equal to interpreter fixed displacement; direction/magnitude separately audited."}
    receipt["passed"] &= receipt["source_labels_sha256"]==receipt["source_labels_expected_sha256"]
    rows=[json.loads(x) for s in ("train","validation","test") for x in (args.data/(s+".jsonl")).read_text().splitlines()]
    displacement=defaultdict(list)
    for r in rows:
        token=r["target"];part=token.split("_")[0].lower()
        if part in ("left","right") and len(token.split("_"))==2 and token.split("_")[1] not in ("OPEN","CLOSE"):
            displacement[part].append(float(np.linalg.norm(r["label_evidence"]["delta_eef_base_m"][part])))
        elif part=="torso":
            displacement["torso"].append(float(np.linalg.norm(np.mean(list(r["label_evidence"]["delta_eef_base_m"].values()),axis=0))))
    receipt["displacement_m_quantiles"]={k:{"count":len(v),"q0_q25_q50_q75_q100":np.quantile(v,[0,.25,.5,.75,1]).tolist()} for k,v in displacement.items()}
    write_json(args.data/"frame_audit.json",receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="frame_comparisons"}),flush=True)
    if not receipt["passed"]:raise RuntimeError("Expert frame/label identity gate failed")
    if args.panels:panels(args.data,groups)


def panels(data,groups):
    import av
    from PIL import Image,ImageDraw
    all_rows=[json.loads(x) for s in ("train","validation") for x in (data/(s+"_images.jsonl")).read_text().splitlines()]
    buckets=defaultdict(list)
    for r in all_rows:buckets[(r["task_id"],r["target"].split("_")[0])].append(r)
    selected=[]
    for key,rows in sorted(buckets.items()):
        byclass={}
        for r in sorted(rows,key=lambda r:sha_text(r["id"])):byclass.setdefault(r["target"],r)
        selected.extend(list(byclass.values())[:3])
    selected=selected[:36]
    out=data/"review";out.mkdir(exist_ok=False)
    for i,r in enumerate(selected):
        e=groups[str(r["episode_index"])]["episode"]
        canvas=Image.new("RGB",(768,600),"white");draw=ImageDraw.Draw(canvas)
        title=f'{i:02d} {r["id"]} {r["split"]} {r["target"]}   current -> +16 frames'
        draw.text((8,3),title,fill="black")
        desc=r["active_instruction"]
        for j in range(0,min(len(desc),210),105):draw.text((8,20+j//105*14),desc[j:j+105],fill="black")
        for col,(view,camera) in enumerate(CAMERAS.items()):
            canvas.paste(Image.open(data/r["images"][view]).convert("RGB"),(col*256,54))
            key="videos/observation.rgb."+camera
            path=ROOT/f'{key}/chunk-{e[key+"/chunk_index"]:03d}/file-{e[key+"/file_index"]:03d}.mp4'
            target=float(e[key+"/from_timestamp"])+r["future_frame"]/30
            with av.open(str(path)) as c:
                st=c.streams.video[0];c.seek(int(target/st.time_base),stream=st,backward=True)
                frame=next(f for f in c.decode(st) if float(f.pts*st.time_base)>=target-1/60)
                if abs(float(frame.pts*st.time_base)-target)>1/60+1e-4:raise RuntimeError("Review future frame mismatch")
                canvas.paste(frame.to_image().convert("RGB").resize((256,256)),(col*256,316))
            draw.text((col*256+5,578),view,fill="black")
        name=f'{i:02d}_{r["id"]}_{r["target"]}.png';canvas.save(out/name)
        r["review_panel"]=name
    write_json(out/"selection.json",selected)
    print(json.dumps({"panels":len(selected),"review_path":str(out)}),flush=True)


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=="__main__":main()

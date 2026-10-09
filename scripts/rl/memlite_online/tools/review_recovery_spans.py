"""Exact-anchor three-camera sheets and causal physical evidence for review."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import canonical,digest,file_sha
from recovery_sft_data import CandidateArchiveReader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','audit','queue','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    inventory=json.loads((a.audit/'inventory.json').read_text())
    anchors=[json.loads(x) for x in (a.audit/'anchors.jsonl').read_text().splitlines()]
    by_id={r['sample_id']:r for r in anchors};by_clock={(canonical(r['source_episode']),r['control_step']):r for r in anchors}
    reader=CandidateArchiveReader(a.raw,inventory);spans=json.loads(a.queue.read_text());items=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for i,span in enumerate(spans):
        candidate=by_id[span['sample_ids'][min(1,len(span['sample_ids'])-1)]];t=candidate['control_step']
        raw,_=reader.episode(candidate['source_episode'])
        points=[x for x in (t-16,t,t+16) if (canonical(candidate['source_episode']),x) in by_clock]
        sheet=Image.new('RGB',(len(points)*672,296),'white');draw=ImageDraw.Draw(sheet)
        title=f'{i:02d} {candidate["task"]} instance={candidate["instance_id"]} {candidate["split"]} LABEL TIME={t}'
        draw.text((4,3),title,font=font,fill='black')
        for j,step in enumerate(points):
            other=by_clock[(canonical(candidate['source_episode']),step)];_,images=reader.observation(other)
            draw.text((j*672+4,23),f's[{step}] '+('LABEL ANCHOR' if step==t else 'context only'),font=font,fill='black')
            for c,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                sheet.paste(Image.fromarray(images[camera]),(j*672+c*224,43))
        skills=json.loads(candidate['actor_input']['issued_skills_semantic_json'])
        draw.text((4,272),'; '.join(f'{s["verb"]} {s["target"]} {s["arm"]}' for s in skills),font=font,fill='black')
        name=f'span-{i:02d}.png';sheet.save(a.output/name)
        evidence=dict(sample_id=candidate['sample_id'],source_episode=candidate['source_episode'],source_group=candidate['source_group'],
            label_anchor=t,original_image_references=[by_clock[(canonical(candidate['source_episode']),step)]['actor_input']['rgb'] for step in points],
            members=candidate['label_audit']['outcome']['members'],
            causal_physics=[dict(control_step=s,physical_audit=raw[s]['physical_audit']) for s in range(t-6,t)],
            future_physics_used=False,independent_event_conservative=digest([candidate['source_episode'],
                [(m['entity'],m.get('evidence_arms')) for m in candidate['label_audit']['outcome']['members']]]))
        path=a.output/f'span-{i:02d}-evidence.json';path.write_text(json.dumps(evidence,indent=2)+'\n')
        items.append(dict(ordinal=i,sample_id=candidate['sample_id'],task=candidate['task'],source_group=candidate['source_group'],split=candidate['split'],
            step=t,event_id=evidence['independent_event_conservative'],members=evidence['members'],
            image=dict(path=name,sha256=file_sha(a.output/name)),physics=dict(path=path.name,sha256=file_sha(path)),
            reviewed_start=min(points),reviewed_end=max(points),review_status='pending'))
    (a.output/'index.json').write_text(json.dumps(dict(items=items,inventory_sha256=digest(inventory),
        anchors_sha256=file_sha(a.audit/'anchors.jsonl'),training_approved=False),indent=2)+'\n')
    print(json.dumps(dict(spans=len(items),output=str(a.output))))


if __name__=='__main__':main()

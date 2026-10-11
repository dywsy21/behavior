"""Render only actual intent and raw outcome predictions below a full MP4."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import subprocess
import unicodedata

from render_causal_execution_video import sha,read,jsonl,media_info,minimal_texts,line


def wrap(text, maximum=96):
    lines=[];row='';width=0
    for c in text:
        n=2 if unicodedata.east_asian_width(c) in ('W','F') else 1
        if width+n>maximum:lines.append(row);row='';width=0
        row+=c;width+=n
    if row:lines.append(row)
    return lines


def intervals(chunks, controls, *, expected_steps):
    if len(controls)!=expected_steps or not chunks or chunks[0]['display']['control_step']!=0:
        raise ValueError('Incomplete full episode clock')
    starts=[r['display']['control_step'] for r in chunks]
    if starts!=list(range(0,expected_steps,16)):raise ValueError('Missing/repeated action chunks')
    for step,row in enumerate(controls):
        chunk=chunks[step//16]
        if (row['control_step']!=step or row['simulator_apply_ack'] is not True or row['token']!=chunk['token']
                or row['action_executed_raw23']!=chunk['actions'][step%16]):
            raise ValueError('Video control does not match actual action ACK')
        if bool(row['terminated'] or row['truncated'])!=(step==expected_steps-1):
            raise ValueError('Missing terminal or controls continued after terminal')
    result=[]
    for i,chunk in enumerate(chunks):
        display=chunk['display'];observer=display['observer']
        if observer['control_step']>display['control_step'] or observer['attempt_generation']!=display['attempt_generation']:
            raise ValueError('Future/foreign observer cannot subtitle this action')
        texts=minimal_texts(display['goal'],observer['members'],display['decision'])
        # Frame zero is s[0]. Every other frame is s[t+1], so the next
        # planning event at t=16 belongs first to frame17, never frame16.
        begin=0 if i==0 else starts[i]+1
        end=starts[i+1]+1 if i+1<len(starts) else expected_steps+1
        result.append(dict(begin=begin,end=end,texts=texts,token=chunk['token']))
    return result


def timestamp(frame):
    cs=frame*100//30
    return f'{cs//360000}:{cs//6000%60:02d}:{cs//100%60:02d}.{cs%100:02d}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--service',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    status=read(a.episode/'status.json');service=read(a.service/'result.json')
    if (status['status']!='completed' or service['status']!='completed' or not service['weights_unchanged']
            or status['config_sha256']!=service['config_sha256'] or status['case']!=service['case']
            or status['source_commit']!=service['source_commit'] or service['optimizer_updates']!=0
            or status['controls']!=service['result']['control_step']):
        raise ValueError('Full unchanged-model episode and matching server completion required')
    hello=read(a.episode/'service-hello.json');events=jsonl(a.service/'events.jsonl')
    if (events[0]['kind']!='begin' or events[-1]['kind']!='close' or events[0]['identity']!=hello['identity']
            or any(r['episode']!=hello['identity']['episode'] for r in events)):
        raise ValueError('Missing/foreign complete server session')
    chunks=jsonl(a.episode/'chunks.jsonl');controls=jsonl(a.episode/'controls.jsonl')
    actions=[r for r in events if r['kind']=='action']
    if len(actions)!=len(chunks):raise ValueError('Server/client action logs disagree')
    for action,chunk in zip(actions,chunks):
        if (action['token']!=chunk['token'] or action['display']!=chunk['display']
                or action['raw_actions']!=chunk['actions']):raise ValueError('Invented/misaligned model output')
    pieces=intervals(chunks,controls,expected_steps=status['controls'])
    sources=list((a.episode/'videos').glob('*.mp4'))
    if len(sources)!=1:raise ValueError('Exactly one source full video required')
    source=sources[0];info=media_info(source)
    if (info['width'],info['height'],info['r_frame_rate'],int(info['nb_read_frames']))!=(672,448,'30/1',status['controls']+1):
        raise ValueError('Official initial frame plus all real controls required')
    rows=max(sum(len(wrap(text)) for text in r['texts']) for r in pieces)
    height=896+24+36*rows+20; height+=height%2
    a.output.mkdir(parents=True)
    ass=['[Script Info]','ScriptType: v4.00+','PlayResX: 1344',f'PlayResY: {height}','WrapStyle: 2',
        '[V4+ Styles]','Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding',
        'Style: Default,Noto Serif CJK SC,25,&H00EEEEEE,&H00EEEEEE,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1',
        '[Events]','Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text']
    for piece in pieces:
        y=910
        for category,text in enumerate(piece['texts']):
            for part in wrap(text):
                ass.append(f"Dialogue: 0,{timestamp(piece['begin'])},{timestamp(piece['end'])},Default,,0,0,0,,"+
                    line(y,part,size=25,color='FFFF8A' if category==0 else 'E8E8E8'))
                y+=36
    subtitle=a.output/'overlay.ass';subtitle.write_text('\n'.join(ass)+'\n')
    if any(c in str(subtitle.resolve()) for c in "':,[]\\"):raise ValueError('Unsupported filter path')
    output=a.output/'execution.mp4'
    cmd=['ffmpeg','-nostdin','-v','warning','-n','-i',str(source),'-an','-vf',
        f'scale=1344:896:flags=lanczos,pad=1344:{height}:0:0:color=0x101722,ass=filename={subtitle.resolve()}',
        '-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(cmd,check=True)
    final=media_info(output)
    if final['nb_read_frames']!=info['nb_read_frames'] or final['r_frame_rate']!='30/1':raise ValueError('Video clock changed')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(output),'-f','null','-'],check=True)
    manifest=dict(status='full_episode_encoded_and_decoded',source_sha256=sha(source),output_sha256=sha(output),
        simulator_status_sha256=sha(a.episode/'status.json'),service_result_sha256=sha(a.service/'result.json'),
        service_events_sha256=sha(a.service/'events.jsonl'),client_chunks_sha256=sha(a.episode/'chunks.jsonl'),
        control_ack_sha256=sha(a.episode/'controls.jsonl'),stream=final,playback_speed=1,
        all_original_frames_preserved=True,initial_frame=True,ending_freeze=False,actual_steps=status['controls'],
        official_result=status['official_result'],model_identity=service['model_identity'],
        probability_meaning='raw softmax of actual online per-member outcome logits, not physical success probability',
        caption_fields=['current_intent','outcome_prediction_and_probability'],owner_visual_review_pending=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (a.output/'timeline.json').write_text(json.dumps(pieces,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()

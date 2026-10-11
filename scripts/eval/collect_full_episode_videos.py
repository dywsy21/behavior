"""Collect and render the three already-running, explicitly named video jobs.

Read-only SSH/SCP of completed evidence; never logs in to VPN, launches a
simulator, retries an episode, changes weights, or touches active source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PAIRS={'radio':('radio-v2','radio-v3'), 'tripod':('tripod-v1','tripod-v2'), 'food':('food-v2','food-v3')}
RTX='/run/ti/rl_memlite_stage1_20261006/runs/full_episode_videos_20261011'
A800='/data/workspace/wsy/behavior2026/runs/full_episode_videos_20261011'
SSH_RTX=['ssh','-p','10383','-o','BatchMode=yes','-o','ConnectTimeout=15','root@42.192.34.154']
SSH_A800=['ssh','-F','/home/wsy/.ssh/lc-a800.conf','-S','/home/wsy/.ssh/lc1-full-video-20261011.sock',
          '-o','BatchMode=yes','-o','ConnectTimeout=15','lc1']


def run(argv):
    # Only re-read status after a transport failure. Never re-run an episode,
    # copy into a partial destination, reconnect VPN, or retry a renderer error.
    for attempt in range(3):
        try:
            return subprocess.check_output(argv,text=True,timeout=60)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
            if (isinstance(error,subprocess.CalledProcessError) and error.returncode!=255) or attempt==2:
                raise
            time.sleep(5)


def prepare_output(output,resume):
    if not resume:
        output.mkdir(parents=True,exist_ok=False)
        return dict(status='waiting_for_original_runs',started=time.time(),completed={},cases={})
    previous=output/'collector-status.json'
    original=previous.read_text()
    state=json.loads(original)
    if state['status']!='collector_failed_original_runs_untouched':
        raise ValueError('Resume is only allowed after the original collector has failed')
    if set(state['completed'])-set(PAIRS) or set(state['cases'])-set(PAIRS):
        raise ValueError('Unexpected episode in saved collector state')
    for case,receipt in state['completed'].items():
        video=(output/case/'execution.mp4').resolve()
        manifest=json.loads((output/case/'manifest.json').read_text())
        if (Path(receipt['video']).resolve()!=video
                or manifest['status']!='full_episode_encoded_and_decoded'
                or hashlib.sha256(video.read_bytes()).hexdigest()!=receipt['sha256']
                or manifest['output_sha256']!=receipt['sha256']
                or manifest['official_result']!=receipt['official_result']):
            raise ValueError('Completed video differs from its original receipt')
    # Partial copy/render destinations need explicit inspection, never overwrite.
    for case in set(PAIRS)-set(state['completed']):
        if any(p.exists() for p in (output/case,output/'source'/case,output/(case+'-render.log'))):
            raise ValueError('Partial collection exists; preserve it for owner inspection')
    with (output/f'collector-status.before-resume-{time.time_ns()}.json').open('x') as archive:
        archive.write(original)
    state.pop('error',None)
    state.update(status='waiting_for_original_runs',resumed=time.time())
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Resume failed read-only collection; never retry simulations')
    a=p.parse_args()
    state=prepare_output(a.output,a.resume)
    def publish():
        state['updated']=time.time()
        tmp=a.output/'collector-status.tmp'
        tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        tmp.replace(a.output/'collector-status.json')
    publish()
    try:
        while len(state['completed'])<len(PAIRS):
            for case,(sim,model) in PAIRS.items():
                if case in state['completed']:continue
                ss=json.loads(run(SSH_RTX+['cat',f'{RTX}/{sim}/status.json']))
                ms=json.loads(run(SSH_A800+['cat',f'{A800}/{model}/status.json']))
                state['cases'][case]=dict(sim_status=ss['status'],model_status=ms['status'],
                    controls=ss.get('controls',0),official_max_steps=ss.get('official_max_steps'))
                publish()
                if 'failed' in ss['status'] or 'failed' in ms['status']:
                    raise RuntimeError(f'{case} failed; preserve original evidence and do not replay')
                if ss['status']!='completed' or ms['status']!='completed':continue
                destination=a.output/'source'/case;destination.mkdir(parents=True,exist_ok=False)
                subprocess.run(['scp','-O','-P','10383','-r',f'root@42.192.34.154:{RTX}/{sim}',
                    str(destination/'episode')],check=True)
                service=destination/'service';service.mkdir()
                for name in ('status.json','result.json','events.jsonl'):
                    subprocess.run(['scp','-F','/home/wsy/.ssh/lc-a800.conf','-o',
                        'ControlPath=/home/wsy/.ssh/lc1-full-video-20261011.sock',
                        f'lc1:{A800}/{model}/{name}',str(service/name)],check=True)
                target=a.output/case
                with (a.output/(case+'-render.log')).open('x') as log:
                    subprocess.run([sys.executable,str(Path(__file__).with_name('render_full_episode_video.py')),
                        '--episode',str(destination/'episode'),'--service',str(service),'--output',str(target)],
                        check=True,stdout=log,stderr=subprocess.STDOUT)
                manifest=json.loads((target/'manifest.json').read_text())
                state['completed'][case]=dict(video=str((target/'execution.mp4').resolve()),
                    sha256=manifest['output_sha256'],official_result=manifest['official_result'],
                    owner_visual_review_pending=True)
                publish();print(json.dumps(dict(event='video_ready_for_owner_review',case=case,**state['completed'][case])),flush=True)
            if len(state['completed'])<len(PAIRS):time.sleep(30)
        state['status']='all_three_videos_rendered_pending_owner_review';publish()
    except BaseException as e:
        state.update(status='collector_failed_original_runs_untouched',error=repr(e));publish();raise


if __name__=='__main__':main()

"""One persistent official TRAIN environment, one GPU, private Kit runtime."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from multiprocessing.connection import Client
import os
import sys
import time
import traceback
import numpy as np

from common import REPO, OUT, RUNTIME, ADAPTER, TEMPLATE, sha, send, recv, save


def main(worker, port, pool=None):
    manifest=json.loads((OUT/'manifest.json').read_text())
    spec=(manifest['workers'] if pool is None else manifest['sim_pools'][pool])[worker]
    from g05.rl.protocol import validate_worker,validate_phase
    evaluation=pool in ('baseline','final')
    validate_worker(spec,evaluation=evaluation)
    if spec['worker'] != worker: raise ValueError('Wrong worker identity')
    cores=set(range(72+8*worker,80+8*worker))
    if not cores <= os.sched_getaffinity(0): raise ValueError('Registered CPU affinity unavailable')
    os.sched_setaffinity(0,cores)
    out=(OUT if pool is None else OUT/pool)/f'worker_{worker}'; out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(REPO/'scripts/semantic_robot'))
    import native_rl_profile
    from rl_reset_boundary import close_before_reset
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1 import official_factory as factory
    from semantic_robot.v2.synchronous_io import native_adapter
    from semantic_robot.v2.render_batch import OwnedFrameRegistration
    from semantic_robot.v2.onboard import OnboardRGBD
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    if not evaluation and sha(spec['actions'])!=spec['actions_sha256']: raise ValueError('Changed TRAIN controls')
    if evaluation and sha(spec['instance_file'])!=spec['instance_file_sha256']:
        raise ValueError('Changed official evaluation instance')
    window=replace(factory.load_official_oracle_window(TEMPLATE),task_name=spec['task'],
                   official_mode=spec['split'],instance_id=spec['instance'],seed=spec['seed'],max_steps=None)
    conn=Client(('127.0.0.1',port),authkey=bytes.fromhex(os.environ['BEHAVIOR_RL_IPC_KEY']))
    send(conn,dict(hello=worker,pid=os.getpid()))
    tasks=factory.load_task_instructions(factory.DEFAULT_TASKS_PATH)
    log=(out/'steps.jsonl').open('x',buffering=1)
    iolog=(out/'io.jsonl').open('x',buffering=1)
    controls=episode_controls=0; episode=-1
    frame_registration=OwnedFrameRegistration()
    terminal=success=False; io=None; video=None; current_clock=None
    def array(x): return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    def io_write(row): iolog.write(json.dumps(dict(worker=worker,episode=episode,**row))+'\n')
    @contextmanager
    def owned_session():
        nonlocal video,io
        with native_rl_profile.session(factory,window,gpu=spec['gpu'],output=out,
                runtime=(RUNTIME if pool is None else RUNTIME/pool)/f'worker_{worker}',
                evaluation_only=evaluation) as session:
            try: yield session
            except BaseException as error:
                # Official __exit__ may close Kit and exit Python itself.
                # Persist/send the PRIMARY error before entering that cleanup.
                row=dict(error=repr(error),traceback=traceback.format_exc(),controls=controls,episode=episode)
                save(out/'primary_failure.json',row)
                try: send(conn,row)
                except BaseException: pass
                raise
            finally:
                if video is not None: video.close(); video=None
                if io is not None:
                    if sys.exc_info()[0] is not None: io.abandon_read_after_failure()
                    io.close(); io=None
    try:
        with owned_session() as session:
            import omnigibson as og
            evaluator=session.evaluator; env=evaluator.env
            reader=OnboardRGBD(env)

            def observe(tag):
                nonlocal current_clock
                capture=io.synchronize(); images={}; receipts={}
                for view,sensor in reader.sensors.items():
                    obs,_=sensor.get_obs()
                    rgb=array(obs['rgb'])[...,:3].copy(); depth=array(obs['depth_linear']).copy()
                    images[view+'_rgb']=rgb.transpose(2,0,1)
                    receipts[view]=dict(native_time=capture[view],rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())
                io.verify_read(receipts)
                obs,_=env.get_obs(); evaluator.obs=evaluator._preprocess_obs(obs)
                raw=factory.behavior_obs_to_native_low(evaluator.obs,tasks)
                raw['images']=images
                raw={k:raw[k] for k in ('images','state','task','embodiment_type','frequency')}
                current_clock=io.clock()
                canvas=Image.new('RGB',(960,768),'black')
                canvas.paste(Image.fromarray(images['head_rgb'].transpose(1,2,0)).resize((720,720)),(0,48))
                for i,key in enumerate(('left_wrist_rgb','right_wrist_rgb')):
                    canvas.paste(Image.fromarray(images[key].transpose(1,2,0)).resize((240,240)),(720,48+i*240))
                ImageDraw.Draw(canvas).text((8,12),f'RL G05-50k {spec["split"]} {spec["instance"]} | ep {episode} | {tag} | episode control {episode_controls} | total {controls}',fill='white')
                video.append_data(np.asarray(canvas))
                if tag in ('reset','curriculum_start'): canvas.save(out/f'{tag}_{episode:03d}.png')
                held={arm:obj.name if obj is not None else None for arm,obj in env.robots[0]._ag_obj_in_hand.items()}
                # Privileged diagnostics remain OUTSIDE the actor observation.
                return dict(observation=raw,clock=current_clock,held=held,success=success,terminal=terminal,
                            episode=episode,episode_controls=episode_controls,total_controls=controls)

            def reset():
                nonlocal io,episode,episode_controls,terminal,success,video
                if io is not None:
                    close_before_reset(io,og.sim,env.robots[0],io_write); io=None
                if video is not None: video.close(); video=None
                session.reset()
                episode+=1; episode_controls=0; terminal=success=False
                io=native_adapter(og.sim,reader.sensors,io_write,registration=frame_registration)
                video=imageio.get_writer(out/f'episode_{episode:03d}.mp4',fps=30/16,codec='libx264',quality=7,macro_block_size=None)
                result=observe('reset')
                if pool is not None:
                    physical={}
                    for obj in sorted(env.scene.objects,key=lambda x:x.name):
                        pos,quat=obj.get_position_orientation()
                        physical[obj.name]=np.concatenate([array(pos).ravel(),array(quat).ravel()]).astype(float).tolist()
                    robot=env.robots[0]
                    physical['__robot_joints__']=np.concatenate([array(robot.get_joint_positions()).ravel(),
                        array(robot.get_joint_velocities()).ravel()]).astype(float).tolist()
                    if not all(np.isfinite(v).all() for v in physical.values()): raise ValueError('Invalid reset state')
                    save(out/f'reset_state_{episode:03d}.json',physical)
                    result['reset_state']=physical
                return result

            send(conn,dict(ready=True,worker=worker,pid=os.getpid(),**reset()))
            while True:
                command=recv(conn,timeout=7200)
                op=command['op']
                if op=='close': send(conn,dict(closed=True,total_controls=controls)); break
                if op=='reset': send(conn,reset()); continue
                if op=='observe': send(conn,observe(command.get('tag','observation'))); continue
                if op!='step' or terminal: raise ValueError('Only live step/reset/observe/close commands allowed')
                if pool is not None: validate_phase(spec,command['phase'])
                actions=np.array(command['actions'],dtype=np.float32,copy=True)
                if actions.ndim!=2 or actions.shape[1]!=23 or not 1<=len(actions)<=16 or not np.isfinite(actions).all():
                    raise ValueError('Invalid action chunk')
                if io.clock()!=current_clock: raise ValueError('Physics advanced while waiting for learner')
                rewards=[]; terminated=truncated=False; start=time.monotonic()
                for action in actions:
                    with io.control(render_requested=False):
                        _,_,terminated,truncated,info=env.step(action,n_render_iterations=1)
                    controls+=1; episode_controls+=1
                    won=bool(info.get('done',{}).get('success',False))
                    rewards.append(float(won and not success)); success=success or won
                    terminal=bool(terminated or truncated)
                    if won and not terminated: raise RuntimeError('Official success without termination')
                    held={arm:obj.name if obj is not None else None for arm,obj in env.robots[0]._ag_obj_in_hand.items()}
                    log.write(json.dumps(dict(control=controls,episode=episode,phase=command['phase'],
                        episode_control=episode_controls,action=action.tolist(),success=won,
                        terminated=bool(terminated),truncated=bool(truncated),held=held))+'\n')
                    if terminal: break
                result=observe(command['phase'])
                result.update(rewards=rewards,terminated=bool(terminated),truncated=bool(truncated),
                              actual_controls=len(rewards),seconds=time.monotonic()-start)
                send(conn,result)
    except BaseException as error:
        row=dict(error=repr(error),traceback=traceback.format_exc(),controls=controls,episode=episode)
        save(out/'failure.json',row)
        try: send(conn,row)
        except BaseException: pass
        raise
    finally:
        if video is not None: video.close()
        if io is not None:
            if sys.exc_info()[0] is not None: io.abandon_read_after_failure()
            io.close()
        log.close(); iolog.close(); conn.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--worker',type=int,choices=(0,1),required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--pool',choices=('baseline','training','final')); args=p.parse_args()
    main(args.worker,args.port,args.pool)

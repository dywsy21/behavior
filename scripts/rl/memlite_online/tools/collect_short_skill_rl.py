"""Repeated legal cold starts for the shared A800 short-skill PPO service.

Physics/reward evidence stays on the simulator side of the policy boundary.
Each physical control and actual early tail is ACKed; a timeout is UNKNOWN,
never an invented failure. Only pinned original TRAIN starts are admitted.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'scripts/eval/memlite_sft100'),str(Path(__file__).resolve().parents[1]/'code'),str(REPO/'src')]
from common import atomic_json,OFFICIAL,OFFICIAL_COMMIT,sha256
from recovery_corpus import digest,group_key,split_group
from recovery_recorder import proprio61
from behavior_branch_state import restore_branch_metadata
from skill_aligned_reward import SkillIdentity,SkillReward,validate_placement_start
from skill_sim_measurements import OmniSkillMeasurements,vector
from recovery_gpu_ownership import owns_short_skill_auxiliary,owns_collection_auxiliary,declared_collection_peers,require_owned_gpu_inventory
from skill_observation_archive import SkillObservationArchive
from skill_training_protocol import rollout_end_control
from wire import packb,unpackb


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--case',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--port',type=int,required=True)
    ap.add_argument('--single-episode',action='store_true',
        help='Exit after one ACKed episode; an external worker starts a fresh simulator, never retries a failed start')
    ap.add_argument('--peer-collection',type=Path)
    ap.add_argument('--peer-collection-commit')
    a=ap.parse_args();cfg=json.loads(a.config.read_text());case=cfg['cases'][a.case];source_spec=case['sim']
    training_multiplier=cfg.get('training_rollout_multiplier',1)
    rollout_end_control(case,'train',training_multiplier)
    if bool(a.peer_collection)!=bool(a.peer_collection_commit):
        raise ValueError('An explicit collection directory and its frozen commit must be supplied together')
    if a.peer_collection and (not (a.peer_collection/'status.json').is_file()
            or len(a.peer_collection_commit)!=40
            or any(c not in '0123456789abcdef' for c in a.peer_collection_commit)):
        raise ValueError('Invalid frozen collection peer')
    collection_peers=declared_collection_peers(cfg.get('collection_peers',[]),
        legacy_path=a.peer_collection,legacy_commit=a.peer_collection_commit)
    if cfg.get('schema')!='short_skill_rl_a800_v1' or cfg.get('user_goal_authorized') is not True:
        raise ValueError('Explicit short-skill goal recipe required')
    archive_protocol=cfg.get('observation_archive','disabled')
    if archive_protocol not in ('disabled','lossless_chunk_boundaries_v1'):
        raise ValueError('Unknown exact-observation archival contract')
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New output and clean frozen source required')
    official=json.loads((REPO/'scripts/eval/memlite_sft100/official_manifest.json').read_text())
    if official['commit']!=OFFICIAL_COMMIT or any(sha256(OFFICIAL/k)!=v for k,v in official['files'].items()):
        raise ValueError('Unpinned installed simulator')
    proof_path=Path(source_spec['proof'])
    if sha256(proof_path)!=source_spec['proof_sha256']:raise ValueError('Changed cold-start physical QA')
    proof=json.loads(proof_path.read_text())
    if proof['task']!=case['task'] or proof['instance_id']!=case['instance_id'] or proof['split']!='train':
        raise ValueError('Proof belongs to another instance/split')
    protected=json.loads(Path(cfg['sim_protected_groups']).read_text())
    group=group_key(case['task'],case['instance_id'])
    if group in protected['groups'] or split_group(case['task'],case['instance_id'])!='train':
        raise ValueError('Never learn on held-out or recovery-DEV starts')
    import numpy as np
    import torch
    import imageio.v2 as iio
    from PIL import Image
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosedOK
    directory=Path(source_spec['directory']);kind=source_spec['kind'];begin=case['start_control']
    if kind=='recovery':
        branch=directory/source_spec['branch'];manifest=json.loads((branch/'manifest.json').read_text())
        result=json.loads((directory/'result.json').read_text());source=json.loads((directory/'source.json').read_text())
        plans=json.loads((branch/'plans.json').read_text());rows=[json.loads(x) for x in (branch/'transitions.jsonl').read_text().splitlines()]
        if (proof['status']!='passed_recorded_correction_sensor_reward' or not proof['physical_skill_reward']
                or proof['source_branch_sha256']!=case['manifest_sha256']
                or sha256(branch/'manifest.json')!=case['manifest_sha256']
                or sha256(branch/'transitions.jsonl')!=manifest['transitions_sha256']
                or sha256(branch/'plans.json')!=manifest['plans_sha256']
                or sha256(directory/'full_snapshot.pt')!=result['full_snapshot_sha256']
                or result['full_snapshot_sha256']!=proof['full_snapshot_sha256']
                or source['source_episode']['split']!='train' or source['source_group']!=group
                or result['split']!='train' or len(plans)!=2 or plans[1]['control_step']!=begin
                or plans[1]['event_sha256']!=case['context_id']
                or plans[1]['parent_goal']!=case['parent_goal']
                or plans[1]['active_skills_semantic_json']!=case['semantic_bundle']
                or len(rows)!=case['end_control'] or any(x['simulator_apply_ack'] is not True for x in rows)):
            raise ValueError('Recovery origin/skill/ACK/snapshot differs from the registered proof')
        saved=torch.load(directory/'full_snapshot.pt',weights_only=False,map_location='cpu')
        resolved=directory/'resolved_config.json';reference=np.asarray([x['action_executed_raw23'] for x in rows],np.float32)
        source_proprio=np.asarray(rows[0]['proprio_before'])
    elif kind=='placement':
        result=json.loads((directory/'result.json').read_text());proposal=Path(source_spec['proposal'])
        source=json.loads((proposal/'manifest.json').read_text());segment=source['selected_segment']
        if (proof['status']!='reference_placement_completed' or not proof['cold_restore']
                or proof['restore_barrier_controls']!=1 or result['cold_restore']
                or source['source_episode']['split']!='train' or source['source_group']!=group
                or source['recovery_split']!='train' or source['task']!=case['task']
                or sha256(proposal/'manifest.json')!=case['manifest_sha256']
                or proof['proposal_sha256']!=case['manifest_sha256']
                or sha256(directory/'start.pt')!=result['start_sha256'] or result['start_sha256']!=proof['start_sha256']
                or proof['start_control']!=begin or proof['end_control']!=case['end_control']
                or segment['start']+1!=begin or segment['semantic']!=case['semantic_bundle']
                or segment['parent']!=case['parent_goal']
                or case['context_id']!=digest([group,segment['start'],segment['semantic']])):
            raise ValueError('Placement origin/skill/barrier differs from the registered proof')
        for name,sha in source['files'].items():
            if Path(name).name!=name or sha256(proposal/name)!=sha:raise ValueError('Changed original placement data')
        saved=torch.load(directory/'start.pt',weights_only=False,map_location='cpu')
        reference=np.load(proposal/'prefix.npz',allow_pickle=False)['action'];resolved=directory/'resolved_config.json'
        source_proprio=np.asarray(saved['proprio'])
    else:raise ValueError('Unsupported legal-start source kind')
    if (reference.ndim!=2 or reference.shape[1]!=23 or not np.isfinite(reference).all()
            or not 0<=begin<case['end_control']<=len(reference)):
        raise ValueError('Invalid original raw23 controls')
    # The pinned Isaac/Omni launcher selects a physical GPU and deliberately
    # unsets CUDA_VISIBLE_DEVICES; masking CUDA can disagree with Vulkan's
    # renderer index. Match that existing tested launch contract exactly.
    env=__import__('os').environ;gpu=env.get('OMNIGIBSON_GPU_ID')
    if 'CUDA_VISIBLE_DEVICES' in env or gpu not in tuple(map(str,range(8))):
        raise ValueError('Select one idle owned simulator GPU without displacing other jobs')
    def snapshot_gpu():
        processes=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader,nounits'],text=True)
        return [(int(line.split(',')[0]),float(line.split(',')[1])) for line in processes.splitlines()]
    def owned_gpu_process(pid,memory):
        proc=Path('/proc')/str(pid)
        # Existing Isaac peers create tiny auxiliary contexts on every card.
        # Exempt only this exact recipe/port and proved output/PID/other-GPU
        # ownership; never ignore an unrelated process merely for being small.
        allowed=False
        try:
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            peer_env=dict(x.decode().split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
            if '--output' in argv:
                peer=Path(argv[argv.index('--output')+1])
                if peer.parent.resolve()==a.output.parent.resolve() and (peer/'status.json').is_file():
                    status=json.loads((peer/'status.json').read_text())
                    allowed=owns_short_skill_auxiliary(pid,float(memory),int(gpu),status,peer,argv,peer_env,
                        config_sha256=sha256(a.config),cases=cfg['cases'],port=a.port)
                for binding in collection_peers:
                    if peer.parent.resolve()==Path(binding['collection']) and (peer/'status.json').is_file():
                        status=json.loads((peer/'status.json').read_text())
                        allowed=allowed or owns_collection_auxiliary(pid,float(memory),int(gpu),status,peer,argv,peer_env,
                            **binding)
        except (OSError,ValueError,IndexError):allowed=False
        return allowed
    gpu_ownership_audit=require_owned_gpu_inventory(snapshot_gpu,owned_gpu_process)
    import omnigibson as og
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    a.output.mkdir(parents=True);completed=0;started=time.monotonic()
    receipt=dict(status='loading',case=a.case,pid=__import__('os').getpid(),config_sha256=sha256(a.config),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        proof_sha256=source_spec['proof_sha256'],optimizer_on_this_host=False,actual_controls=0,
        collection_peers=collection_peers,
        gpu_ownership_audit=gpu_ownership_audit,
        fresh_process_per_episode=a.single_episode,
        completed_episodes=0,protected_progress_contract_tested=False,whole_task_sr=False)
    atomic_json(a.output/'status.json',receipt)
    class NoPolicy:
        def reset(self):pass
    class Evaluator(BatchedEvaluator):
        def load_policy(self):return NoPolicy()
        def __exit__(self,exc_type,exc_value,exc_tb):
            # Official simulator teardown may exit the interpreter. Persist
            # truthful final/error receipts BEFORE delegating to it, rather
            # than leaving an old "executing" status behind on failure.
            receipt.update(seconds=time.monotonic()-started,completed_episodes=completed)
            if exc_type is not None:
                receipt.update(status='failed',error=repr(exc_value))
            elif receipt['status'] not in ('server_finished','server_closed_between_episodes','completed_single_episode'):
                receipt.update(status='failed',error='Simulator context ended without a server boundary')
            atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt)
            return super().__exit__(exc_type,exc_value,exc_tb)
    eval_cfg=OmegaConf.create(json.loads(resolved.read_text()));eval_cfg.write_video=False
    bundle=json.loads(case['semantic_bundle'])
    try:
        with Evaluator(eval_cfg) as evaluator:
            evaluator.load_batch({0:case['instance_id']});inst=evaluator.instance_eval_states[0]
            def observable():
                return dict(images={camera+'_rgb':np.ascontiguousarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3].transpose(2,0,1))
                    for camera,key in evaluator.robot_camera_names.items()},proprio=np.asarray(proprio61(inst.obs),dtype=np.float32))
            def snapshot_rgb(obs,path):
                refs={}
                for camera,image in obs['images'].items():
                    p=path/(camera+'.png');Image.fromarray(image.transpose(1,2,0)).resize((224,224)).save(p);refs[p.name]=sha256(p)
                return refs
            with connect('ws://127.0.0.1:'+str(a.port),max_size=32<<20,compression=None,ping_timeout=None) as socket:
                hello=unpackb(socket.recv(timeout=120))
                if hello!=dict(protocol='short_skill_rl_v1',config_sha256=sha256(a.config)):
                    raise ValueError('Different training service/config')
                while True:
                    try:
                        socket.send(packb(dict(op='job',case=a.case)))
                        assignment=unpackb(socket.recv(timeout=1800))
                    except ConnectionClosedOK:
                        receipt['status']='server_closed_between_episodes';break
                    if assignment.get('status')=='finished':receipt['status']='server_finished';break
                    if assignment.get('status')=='wait':time.sleep(1);continue
                    if assignment.get('status')!='job' or assignment['case']!=case:raise ValueError('Unbound job assignment')
                    job=assignment['job'];episode=a.output/(job['phase']+'-'+str(job['round']).zfill(4)+'-'+str(job['seed']))
                    policy_end=rollout_end_control(case,job['phase'],training_multiplier)
                    episode.mkdir(exist_ok=False);episode_start=time.monotonic()
                    receipt.update(status='restoring',job=job,completed_episodes=completed)
                    atomic_json(a.output/'status.json',receipt)
                    identity=SkillIdentity(assignment['session'],case['task'],case['instance_id'],job['id'],case['context_id'],digest(bundle),0)
                    og.sim.load_state(deepcopy(saved['world']),serialized=False);restore_branch_metadata(evaluator,saved['metadata'])
                    random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
                    torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
                    for _ in range(3):og.sim.render()
                    raw,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(raw[0],inst)
                    error=float(np.max(np.abs(np.asarray(proprio61(inst.obs))-source_proprio)))
                    if error>1e-3:raise ValueError('Repeated cold restore proprio drift')
                    sensor=OmniSkillMeasurements(inst.env_accessor,identity,bundle[0],bundle=bundle)
                    if kind=='placement':
                        for key,obj in [('target',sensor.target),('destination',sensor.destination)]:
                            if np.max(np.abs(vector(obj.get_position_orientation()[0])-saved['start_positions'][key]))>.001:
                                raise ValueError('Placement cold object pose drift')
                    reset_indices=range(begin) if kind=='recovery' else [begin-1]
                    with (episode/'restore-controls.jsonl').open('x') as stream:
                        for t in reset_indices:
                            command=np.asarray(reference[t],np.float32)
                            term,trunc,_=evaluator._apply_actions(torch.as_tensor(command)[None],[0]);receipt['actual_controls']+=1
                            stream.write(json.dumps(dict(control_step=t+1,action_executed_raw23=command.tolist(),simulator_apply_ack=True,
                                reward_assigned=False,actor_input=False,action_source='pinned_restore_not_policy'))+'\n')
                            if bool(term[0]) or bool(trunc[0]):raise ValueError('Native end during legal reset prefix')
                    initial,physical=sensor.read(identity)
                    # Keep exact failed-start evidence BEFORE any comparison
                    # can abort. A failed reset must not be reported as the
                    # preceding completed job or retried until it happens to pass.
                    reset_evidence=dict(job=job,proprio_error=error,restore_controls=len(reset_indices),
                        initial_physical_evidence=physical,fresh_process_per_episode=a.single_episode,
                        target_position=vector(sensor.target.get_position_orientation()[0]).tolist(),
                        original_rgb=snapshot_rgb(observable(),episode))
                    if kind=='recovery':
                        reset_evidence['expected_target_position']=rows[begin]['physical_before']['position']
                        reset_evidence['target_position_max_error']=float(np.max(np.abs(
                            vector(sensor.target.get_position_orientation()[0])-rows[begin]['physical_before']['position'])))
                    atomic_json(episode/'restore-diagnostics.json',reset_evidence)
                    if kind=='placement':validate_placement_start(initial,physical,[],cold=True)
                    else:
                        if np.max(np.abs(vector(sensor.target.get_position_orientation()[0])-rows[begin]['physical_before']['position']))>.005:
                            raise ValueError('Repeated fault target position drift')
                        if 'directed_open_fraction' in physical and abs(physical['directed_open_fraction']-rows[begin]['physical_before']['directed_open_fraction'])>.005:
                            raise ValueError('Repeated fault joint clearance drift')
                    reward=SkillReward(identity,initial,control_step=begin);current=observable();t=begin;video=None
                    atomic_json(episode/'start.json',dict(job=job,identity=asdict(identity),policy_version=assignment['policy_version'],
                        policy_sha256=assignment['policy_sha256'],initial_physical_evidence=physical,proprio_error=error,
                        restore_controls=len(reset_indices),original_rgb=snapshot_rgb(current,episode)))
                    socket.send(packb(dict(op='begin',job_id=job['id'],initial_evidence=physical)))
                    ack=unpackb(socket.recv(timeout=1800))
                    if ack.get('status')!='begun' or ack['identity']!=asdict(identity):raise ValueError('Begin identity mismatch')
                    archive=None
                    if archive_protocol=='lossless_chunk_boundaries_v1':
                        archive=SkillObservationArchive(episode/'observations',identity,
                            policy_version=assignment['policy_version'],policy_sha256=assignment['policy_sha256'],start_control=begin)
                        archive.append(identity,t,current)
                    receipt.update(status='executing',job=job,completed_episodes=completed);atomic_json(a.output/'status.json',receipt)
                    video=iio.get_writer(str(episode/'policy.mp4'),fps=15,codec='libx264',macro_block_size=None)
                    try:
                        with (episode/'controls.jsonl').open('x',buffering=1) as log:
                            while t<policy_end:
                                common=dict(identity=asdict(identity),policy_version=assignment['policy_version'],policy_sha256=assignment['policy_sha256'],control_step=t)
                                socket.send(packb(dict(op='action',observation=current,**common)))
                                result=unpackb(socket.recv(timeout=1800))
                                if (any(result.get(k)!=v for k,v in common.items()) or result['action_chunk'].shape!=(16,23)
                                        or not np.isfinite(result['action_chunk']).all()):raise ValueError('Wrong action/session/clock')
                                controls=[]
                                for command in result['action_chunk'][:result['max_controls']]:
                                    term,trunc,_=evaluator._apply_actions(torch.as_tensor(command.copy())[None],[0]);t+=1;receipt['actual_controls']+=1
                                    measurement,physical=sensor.read(identity)
                                    last=reward.advance(identity,t,measurement,protected_values={},official_terminal=bool(term[0]),
                                                        time_limit=bool(trunc[0]) or t==policy_end)
                                    control=dict(control_step=t,simulator_apply_ack=True,action_executed_raw23=command.tolist(),
                                        physical_evidence=physical,official_terminal=bool(term[0]),official_truncated=bool(trunc[0]))
                                    controls.append(control);current=observable()
                                    log.write(json.dumps(dict(control,skill_reward=dict(last,identity=asdict(identity)),
                                        experience_id=result['experience_id'],policy_version=assignment['policy_version'],action_source=job['phase']))+'\n')
                                    if (t-begin)%2==0:
                                        video.append_data(np.concatenate([np.asarray(Image.fromarray(current['images'][k+'_rgb'].transpose(1,2,0)).resize((224,224)))
                                                                         for k in ('head','left_wrist','right_wrist')],axis=1))
                                    if last['terminated'] or last['truncated']:break
                                socket.send(packb(dict(op='ack',controls=controls,observation=current,observation_control_step=t,**common)))
                                response=unpackb(socket.recv(timeout=1800))
                                if 'error' in response:raise RuntimeError('A800 rejected update: '+response['error'])
                                if (response.get('status')!='acknowledged' or response['control_step']!=t
                                        or response['identity']!=asdict(identity) or response['final_reward']!=dict(last,identity=asdict(identity))
                                        or response['ended']!=(last['terminated'] or last['truncated'])):
                                    raise ValueError('A800 reward/GAE ACK differs from actual simulator measurements')
                                if archive is not None:archive.append(identity,t,current)
                                if response['ended']:break
                    finally:
                        if video is not None:video.close()
                    completed+=1
                    final=dict(job=job,identity=asdict(identity),policy_version=assignment['policy_version'],
                        policy_sha256=assignment['policy_sha256'],skill_success=last['skill_success'],outcome=last['outcome'],
                        scored_controls=t-begin,restore_controls=len(reset_indices),seconds=time.monotonic()-episode_start,
                        reference_end_control=case['end_control'],policy_end_control=policy_end,
                        controls_sha256=sha256(episode/'controls.jsonl'),optimizer_on_this_host=False,whole_task_sr=False)
                    if archive is not None:
                        final['observation_archive']=dict(protocol=archive_protocol,
                            manifest_sha256=sha256(archive.root/'manifest.json'),
                            observations_sha256=sha256(archive.root/'observations.jsonl'))
                    atomic_json(episode/'result.json',final);receipt.update(status='between_episodes',completed_episodes=completed)
                    atomic_json(a.output/'status.json',receipt)
                    if a.single_episode:
                        receipt['status']='completed_single_episode'
                        break
        receipt.update(seconds=time.monotonic()-started,completed_episodes=completed)
        atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt)
    except BaseException as error:
        atomic_json(a.output/'status.json',dict(receipt,status='failed',error=repr(error),seconds=time.monotonic()-started));raise


if __name__=='__main__':main()

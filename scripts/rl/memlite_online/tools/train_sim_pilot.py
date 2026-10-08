from pathlib import Path
import sys,os,json,time
from bootstrap import bootstrap
bootstrap()
ROOT=Path('/run/ti/rl_memlite_stage1_20261006');RUN=Path(os.environ['RL_PILOT_RUN'])
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'));sys.path.append('/run/ti/behavior_stage3_20260930/tools')
import torch,isaacsim
original_app=isaacsim.SimulationApp
isaacsim.SimulationApp=lambda cfg,*a,**k:original_app(dict(cfg,active_gpu=int(os.environ['RL_GPU']),physics_gpu=int(os.environ['RL_GPU']),multi_gpu=False),*a,**k)
from omnigibson.eval.evaluator import BatchedEvaluator
from omnigibson.eval.eval import main
from goal_option_index import install_goal_mask_index
install_goal_mask_index()
from sim_reward_adapter import PhysicalRewardAdapter
from pilot_policy import PilotPolicy
SHARED=os.environ.get('RL_REWARD_PROTOCOL')=='shared_terminal_q_v1'
READ_ONLY=os.environ.get('RL_EVALUATION_ONLY')=='1'
if READ_ONLY:
 from readonly_policy import ReadOnlyPolicy
if SHARED:
 from shared_physical_reward import SharedPhysicalReward
 from shared_pilot_policy import SharedPilotPolicy
 if os.environ.get('RL_SYNCHRONOUS')=='1':
  from distributed_pilot_policy import DistributedPilotPolicy
  SharedPilotPolicy=DistributedPilotPolicy
TASK=sys.argv[sys.argv.index('--task-name')+1]
load=BatchedEvaluator.load_batch;step=BatchedEvaluator._step_fn;run=BatchedEvaluator.run
apply_actions=BatchedEvaluator._apply_actions
load_env=BatchedEvaluator.load_env
BatchedEvaluator.load_policy=lambda self:(ReadOnlyPolicy if READ_ONLY else SharedPilotPolicy if SHARED else PilotPolicy)(self.num_envs,TASK,RUN)

def wrapped_load_env(self,*args,**kwargs):
 if os.environ.get('RL_TASK_FULL_SCENE')=='1':self.cfg.partial_scene_load=False
 return load_env(self,*args,**kwargs)
def wrapped_load(self,*args,**kwargs):
 from state_mirror_compat import install_state_mirror_barrier
 install_state_mirror_barrier(RUN)
 result=load(self,*args,**kwargs)
 self.reward_adapters=[] if READ_ONLY else [SharedPhysicalReward(state.env_accessor) if SHARED else PhysicalRewardAdapter(state.env_accessor,TASK) for state in self.instance_eval_states]
 for adapter in self.reward_adapters:adapter.reset()
 if not READ_ONLY:self.policy.begin_recording(self)
 (RUN/'scene.ready').write_text(json.dumps(dict(pid=os.getpid(),ready=time.time())))
 # Preparation-only barrier: keep all eight real scenes resident before
 # releasing inference, so staggered single-card successes cannot pass as an
 # eight-card concurrency test. The server supplies an absolute deadline.
 if os.environ.get('RL_PREP_BARRIER_DIR'):
  barrier=Path(os.environ['RL_PREP_BARRIER_DIR'])
  (barrier/('gpu_'+os.environ['RL_GPU']+'.ready')).write_text(json.dumps(dict(pid=os.getpid(),ready=time.time())))
  while not (barrier/'RELEASE').exists():
   if time.time()>float(os.environ['RL_PREP_DEADLINE'])-600 or (barrier/'ABORT').exists():
    raise TimeoutError('Eight-card preparation barrier was not released')
   time.sleep(1)
 return result
def wrapped_step(self,active):
 self.policy.active=active
 terminated,truncated=step(self,active)
 self.policy.observe(self,active,terminated,truncated)
 return terminated,truncated
def wrapped_apply_actions(self,actions,active):
 actual=actions.detach().cpu().clone()
 result=apply_actions(self,actions,active)
 if not READ_ONLY:
  for index in active:self.policy.recorder.confirm_applied(index,actual[index])
 return result
def wrapped_run(self,*args,**kwargs):
 try:
  result=run(self,*args,**kwargs);self.policy.flush(self)
  (RUN/(TASK+'.completed')).touch();return result
 finally:
  if not READ_ONLY:self.policy.close_recording('collector_exit')
BatchedEvaluator.load_env=wrapped_load_env;BatchedEvaluator.load_batch=wrapped_load;BatchedEvaluator._step_fn=wrapped_step;BatchedEvaluator.run=wrapped_run
BatchedEvaluator._apply_actions=wrapped_apply_actions
try:main()
except Exception as error:
 (RUN/(TASK+'.error.json')).write_text(json.dumps({'error':repr(error)}));raise

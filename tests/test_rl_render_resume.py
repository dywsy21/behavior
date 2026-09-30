from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/semantic_robot')]
from render_batch_fixture import batch_fixture
from g05.rl.render_resume import audit_render_failure,reconcile_render_stop,OLD_ERROR
from g05.rl.dense_resume import resume_state


class RenderResumeTests(unittest.TestCase):
    def fixture(self):
        steps=[];io=[]
        for i in range(1,3):
            steps.append(dict(control=i,episode=0,episode_control=i,phase='policy',action=[0.]*23))
            io.append(dict(kind='control',call=i,episode=0,completed=True,
                before=dict(physics_index=4*(i-1),simulation_time=(i-1)/30),
                after=dict(physics_index=4*i,simulation_time=i/30)))
        clock=io[-1]['after'];batch=batch_fixture();batch['completed']['numerator']+=1
        io.extend([dict(kind='capture',episode=0,completed=False,error=OLD_ERROR,
                        before=clock,after=clock,batch=batch),
                   dict(kind='close',episode=0,completed=True,before=clock,after=clock)])
        return steps,io,dict(error=OLD_ERROR,controls=2,episode=0)

    def test_physical_actions_native_ticks_and_failure_closure_independently_match(self):
        result=audit_render_failure(*self.fixture())
        self.assertEqual(result,dict(controls=2,policy_controls=2,incomplete_episode=0,
                                     incomplete_controls=2,incomplete_policy_controls=2))

    def test_missing_nonfinite_wrong_ticks_bad_camera_and_unknown_crash_refused(self):
        for bad in ('missing','nonfinite','call','ticks','time','close','mixed','not_mismatch','error'):
            with self.subTest(bad=bad):
                steps,io,failure=self.fixture()
                if bad=='missing':steps.pop()
                if bad=='nonfinite':steps[0]['action'][0]=float('nan')
                if bad=='call':io[0]['call']=0
                if bad=='ticks':io[0]['before']['physics_index']=-4
                if bad=='time':io[0]['before']['simulation_time']=float('nan')
                if bad=='close':io[-1]['completed']=False
                if bad=='mixed':io[-2]['batch']['cameras']['head']['frame']['rationalTimeOfSimNumerator']-=1
                if bad=='not_mismatch':io[-2]['batch']['completed']=deepcopy(io[-2]['batch']['scheduled'])
                if bad=='error':failure['error']='different crash'
                with self.assertRaises(ValueError):audit_render_failure(steps,io,failure)

    def test_both_executed_pending_chunks_debited_no_ledger_reset(self):
        counts=[20778,26504]
        closed=dict(exits=[0,0],reported_controls=[],clean=False,pending_controls=32,ledger_controls=47250)
        unresolved=dict(pending={'0':16,'1':16},budget_pending=32)
        status=dict(phase='failed',controls=55708,pending_controls=32)
        result=reconcile_render_stop(closed,counts,unresolved,status,8458)
        self.assertEqual(result['controls'],55740)
        self.assertEqual(result['reconciled_pending_controls'],32)
        self.assertFalse(result['original_close_clean'])
        for changed in [dict(closed,exits=[0,1]),dict(closed,ledger_controls=47249),
                        dict(closed,pending_controls=16),dict(closed,clean=True)]:
            with self.assertRaises(ValueError):reconcile_render_stop(changed,counts,unresolved,status,8458)
        with self.assertRaises(ValueError):reconcile_render_stop(closed,counts,unresolved,status,0)

    def test_fourteen_batch_curriculum_and_173_optimizer_counters_survive(self):
        manifest=dict(max_batches=64,max_training_controls=80000,
            workers=[dict(worker=0,instance=1,prefix_controls=1076,first_recorded_terminal=1364),
                     dict(worker=1,instance=138,prefix_controls=1096,first_recorded_terminal=1192)])
        prefixes=[1076]*7+[980]*5+[884]*2
        wins0=[True,False,True,False,True,True,True,True,False,True,True,True,False,True]
        updates=[4,4,5,6,8,5,5,8,3,5,8,6,6,6]
        batches=[]
        for i in range(14):
            batches.append(dict(batch=i,new_actor_updates=updates[i],controls=52032,
                checkpoint=f'/run/ckpt{i}.pt',official_reward=int(wins0[i])+int(i==6),nonzero_shaping_controls=1,
                episodes=[dict(worker=0,instance=1,prefix=prefixes[i],success=wins0[i],
                               next_prefix=prefixes[min(i+1,13)]),
                          dict(worker=1,instance=138,prefix=1096,success=i==6,next_prefix=1096)]))
        receipt=dict(path='/run/ckpt13.pt',controls=52032,actor_updates=173,critic_updates=72)
        result=resume_state(manifest,batches,receipt,55740)
        self.assertEqual(result['prefixes'],{'0':884,'1':1096})
        self.assertEqual(result['recent']['0'],[False,True])
        self.assertEqual(result['recent']['1'],[i==6 for i in range(14)])
        self.assertEqual((result['batches'],result['successes'],result['actor_updates'],result['controls']),
                         (14,11,173,55740))


if __name__=='__main__':unittest.main()

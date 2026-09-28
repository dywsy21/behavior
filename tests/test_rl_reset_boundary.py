from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/semantic_robot'))
from rl_reset_boundary import close_before_reset


class ResetBoundaryTests(unittest.TestCase):
    def test_invalidated_views_are_refreshed_without_physics_step(self):
        class Robot:
            valid=True
            _ag_obj_in_hand={'left':None,'right':None}
            def get_joint_positions(self):
                if not self.valid: raise AttributeError('invalid articulation view')
                return np.array([1.,2.])
            def get_joint_velocities(self): return np.zeros(2)
        robot=Robot()
        class IO:
            def clock(self): return dict(simulation_time=1.,physics_index=120)
            def close(self): robot.valid=False
        class Sim:
            current_time=1.; current_time_step_index=120
            def is_playing(self): return True
            def update_handles(self): robot.valid=True
        rows=[]; close_before_reset(IO(),Sim(),robot,rows.append)
        self.assertTrue(robot.valid); self.assertTrue(rows[0]['completed'])
        sim=Sim()
        def unsafe(): robot.valid=True; sim.current_time_step_index+=1
        sim.update_handles=unsafe
        with self.assertRaises(RuntimeError): close_before_reset(IO(),sim,robot,rows.append)


if __name__=='__main__': unittest.main()

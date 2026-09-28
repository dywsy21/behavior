"""Rebind native physics handles after detaching Replicator references.

Detaching a render graph can invalidate Isaac's articulation views although
the simulation is still playing. Scene.restore immediately dumps joint state,
even when batch_remove_objects is empty. Rebind BEFORE calling official reset;
prove that this housekeeping itself does not step or move the robot.
"""
import numpy as np


def snapshot_robot(robot):
    def array(value):
        if hasattr(value,'detach'): value=value.detach().cpu().numpy()
        return np.asarray(value).copy()
    return dict(qpos=array(robot.get_joint_positions()),qvel=array(robot.get_joint_velocities()),
        held={arm:obj.name if obj is not None else None for arm,obj in robot._ag_obj_in_hand.items()})


def close_before_reset(io,sim,robot,record):
    before=io.clock(); state=snapshot_robot(robot)
    io.close()
    if not sim.is_playing(): raise RuntimeError('Graph detach unexpectedly stopped physics')
    sim.update_handles()
    after=snapshot_robot(robot)
    clock=dict(simulation_time=float(sim.current_time),physics_index=int(sim.current_time_step_index))
    errors={key:float(np.max(np.abs(state[key]-after[key]))) for key in ('qpos','qvel')}
    if clock!=before or any(v>1e-6 or not np.isfinite(v) for v in errors.values()) or state['held']!=after['held']:
        raise RuntimeError('Reset handle refresh advanced physics or changed robot state')
    record(dict(kind='reset_handle_rebind',completed=True,before=before,after=clock,
                max_abs_errors=errors,held=after['held']))

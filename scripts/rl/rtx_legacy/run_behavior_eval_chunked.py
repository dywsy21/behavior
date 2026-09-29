"""Run the stock BEHAVIOR evaluator while skipping unused chunk renders.

The G0.5 websocket server predicts an action chunk and serves one cached action
per request.  Only the last environment step in each chunk needs to render a
new camera frame for the next model inference.  Physics, task termination,
metrics, and the official result writer remain unchanged.
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path

import omnigibson as og
from omnigibson.envs import EnvironmentWrapper
from omnigibson.eval import evaluator as evaluator_module
from omnigibson.eval.wrappers import rgbd_full_res_wrapper
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)
from memlite_sim_trace import build_sim_step_payload


ACTION_STEPS = int(os.environ.get("BEHAVIOR_ACTION_STEPS", "16"))
if ACTION_STEPS < 1:
    raise ValueError("BEHAVIOR_ACTION_STEPS must be positive")

_original_reset = evaluator_module.Evaluator.reset
_original_evaluator_init = evaluator_module.Evaluator.__init__
_SIM_TRACE_PATH = os.environ.get("MEMLITE_SIM_TRACE_PATH")
_sim_trace_handle = None


def _record_sim_step(*, action, terminated, truncated, info, phase) -> None:
    """Persist official env.step outputs without recomputing task predicates."""
    global _sim_trace_handle
    if not _SIM_TRACE_PATH:
        return
    try:
        if _sim_trace_handle is None:
            path = Path(_SIM_TRACE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            _sim_trace_handle = path.open("a", encoding="utf-8")
        payload = build_sim_step_payload(
            action=action,
            terminated=terminated,
            truncated=truncated,
            info=info,
            phase=phase,
        )
        _sim_trace_handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        _sim_trace_handle.flush()
    except Exception:
        logging.getLogger(__name__).exception("Failed to write simulator diagnostic trace")


def _rgbd_init_with_deferred_space_reload(self, env) -> None:
    # The stock wrapper reloads the observation space while Evaluator is still
    # constructing the environment.  R1Pro proprioception requires initialized
    # articulation handles, which Evaluator creates only after wrapper setup.
    # Apply the stock sensor configuration here and defer only the space reload.
    EnvironmentWrapper.__init__(self, env=env)
    robot = env.robots[0]
    robot_eval_config = getattr(env, "_eval_robot_config", {})
    camera_roles_by_sensor_name = {
        camera_name.split("::")[1]: camera_id
        for camera_id, camera_name in get_robot_camera_names(
            robot.name, robot_eval_config
        ).items()
    }
    for sensor_name, sensor in robot.sensors.items():
        if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
            continue
        set_sensor_modalities(sensor, {"rgb", "depth_linear"})
        camera_id = camera_roles_by_sensor_name.get(sensor_name)
        if camera_id == "head":
            sensor.image_height, sensor.image_width = HEAD_RESOLUTION
        else:
            sensor.image_height, sensor.image_width = WRIST_RESOLUTION
    self._behavior_deferred_space_reload = True


def _evaluator_init_then_reload_space(self, cfg) -> None:
    _original_evaluator_init(self, cfg)
    if getattr(self.env, "_behavior_deferred_space_reload", False):
        self.env.env.load_observation_space()
        logging.getLogger(__name__).info(
            "Reloaded RGB-D observation space after articulation initialization"
        )


def _chunk_aware_reset(self) -> None:
    _original_reset(self)
    self._behavior_chunk_phase = 0


def _chunk_aware_step(self):
    self.robot_action = self.policy.forward(obs=self.obs)

    phase = getattr(self, "_behavior_chunk_phase", 0)
    render_this_step = phase == ACTION_STEPS - 1
    with og.sim.render_on_step(render_this_step):
        obs, _, terminated, truncated, info = self.env.step(
            self.robot_action, n_render_iterations=1
        )
    obs = self._sync_lights_and_get_obs(obs)
    self.obs = self._preprocess_obs(obs)

    _record_sim_step(
        action=self.robot_action,
        terminated=terminated,
        truncated=truncated,
        info=info,
        phase=phase,
    )

    if self._video_path is not None:
        self._write_video()

    if terminated or truncated:
        self.n_trials += 1
        if info["done"]["success"]:
            self.n_success_trials += 1

    for metric in self.metrics:
        metric.step(
            self.env,
            self.robot_action,
            obs,
            0.0,
            terminated,
            truncated,
            info,
        )

    self._behavior_chunk_phase = (phase + 1) % ACTION_STEPS
    return terminated, truncated


evaluator_module.Evaluator.reset = _chunk_aware_reset
evaluator_module.Evaluator.step = _chunk_aware_step
evaluator_module.Evaluator.__init__ = _evaluator_init_then_reload_space
rgbd_full_res_wrapper.RGBDFullResWrapper.__init__ = (
    _rgbd_init_with_deferred_space_reload
)

logging.getLogger(__name__).warning(
    "Chunk-aware rendering enabled: one render per %d policy actions", ACTION_STEPS
)

from omnigibson.eval.eval import main  # noqa: E402


if __name__ == "__main__":
    main()

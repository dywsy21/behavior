"""Bind every observation and action to its task, environment and policy round."""
import json

import numpy as np

from shared_pilot_policy import SharedPilotPolicy


class DistributedPilotPolicy(SharedPilotPolicy):
    def observations(self, obs, indices):
        for index in indices:
            task_id = int(np.asarray(obs["task_id"][index].detach().cpu()).reshape(-1)[0])
            if task_id != self.episode_metadata[index]["task_index"]:
                raise ValueError("Simulator task ID differs from registered episode")
        result = super().observations(obs, indices)
        for observation, index in zip(result, indices, strict=True):
            observation.update(task_identity=self.task, episode_id=self.episode_metadata[index]["episode_id"])
        return result

    def rpc(self, kind, **payload):
        # Synchronous ranks may wait through another rank's scene reset.
        from a4_wire import packb, unpackb
        self.socket.send(packb(dict(kind=kind, **payload)))
        response = unpackb(self.socket.recv(timeout=1800))
        if not response["ok"]:
            raise RuntimeError(response["error"])
        result = response["response"]
        if kind == "update" and result.get("stop_after_update"):
            # Expected collector exit; unfinished trajectories are not successes.
            (self.run / "shared_training_stopped.json").write_text(json.dumps(dict(
                update=result["update"], reason=result["stop_after_update"])))
            raise SharedTrainingStop(result["stop_after_update"])
        return result


class SharedTrainingStop(RuntimeError):
    pass

"""Task-only AR transport; evaluator subgoals/truth never cross to the actor."""
import asyncio
from collections.abc import Mapping

import numpy as np

OBSERVABLE_FIELDS = {'task', 'embodiment_type', 'frequency', 'images', 'state',
                     'history_action_counts', 'history_is_pad'}


def task_only_request(observation, execute_steps):
    if (not isinstance(observation, dict) or set(observation) != OBSERVABLE_FIELDS
            or type(execute_steps) is not int or execute_steps != 16):
        raise ValueError('Only the seven actual-history fields and sixteen controls are permitted')
    return dict(kind='native_task_chunk', observation=observation, execute_steps=execute_steps)


def validate_native_identity(hello, *, checkpoint_sha, prefix_actions, window_sha):
    identity = hello.get('identity', {})
    if (hello.get('mode') != 'native_task_ar_history6' or hello.get('components') != ['native_task_actor']
            or identity.get('kind') != 'native_task_ar500_prefix_history6_service'
            or identity.get('checkpoint_sha256') != checkpoint_sha
            or identity.get('initial_action_count') != prefix_actions
            or identity.get('prefix_window_sha256') != window_sha
            or identity.get('num_obs_steps') != 6 or identity.get('execution_start') != 0
            or identity.get('max_chunks') != 16 or identity.get('static_format_forced') is not True
            or identity.get('teacher_or_planner_in_actor') is not False):
        raise ValueError('Wrong native-task AR checkpoint/window/format/observation endpoint')
    return identity


def validate_response(response, *, checkpoint_sha, count, anchor_hashes):
    actions = np.asarray(response.get('actions'))
    admission = response.get('history_admission', {})
    if (actions.dtype != np.float32 or actions.shape != (16, 23) or not np.isfinite(actions).all()
            or response.get('actor_identity', {}).get('checkpoint_sha256') != checkpoint_sha
            or response.get('execution_start') != 0 or response.get('execution_steps') != 16
            or response.get('teacher_or_planner_in_actor') is not False
            or response.get('schema', {}).get('complete') is not True
            or response.get('schema', {}).get('rule_safe_clamp') is not False
            or admission.get('consumed_actions') != count or admission.get('execute_steps') != 16
            or admission.get('anchor_hashes') != anchor_hashes):
        raise ValueError('Native AR response changed actions, true history, or policy identity')
    return actions


def make_client(*args, **kwargs):
    """Reuse audited per-physics-step capture; replace the FM protocol entirely."""
    from native_a2_prefix_client import NativeA2PrefixClient
    from native_a2_prefix_history import actual_anchor_hashes

    class NativeTaskARPrefixClient(NativeA2PrefixClient):
        def validate_identity(self, hello):
            return validate_native_identity(hello, checkpoint_sha=self.expected,
                prefix_actions=self.prefix_actions, window_sha=self.window_sha)

        async def _request(self, observation, _installed_evaluator_only, execute_steps):
            from g05.utils.websocket import unpackb
            import websockets
            # _installed_evaluator_only is deliberately not inspected or sent.
            prepared = self.history.request_observation(observation)
            if self.socket is None:
                self.socket = await websockets.connect(self.uri, max_size=128 << 20,
                                                        open_timeout=60, ping_timeout=None)
                hello = unpackb(await asyncio.wait_for(self.socket.recv(), timeout=60))
                self.identity = self.validate_identity(hello)
                reply = await self._call(dict(kind='begin', seed=self.seed))
                if reply.get('seed') != self.seed:
                    raise ValueError('Native AR RNG seed was not bound')
            response = await self._call(task_only_request(prepared, execute_steps))
            actions = validate_response(response, checkpoint_sha=self.expected,
                count=self.history.last_count, anchor_hashes=actual_anchor_hashes(prepared))
            self.requests.append(dict(actual_consumed_actions_before_query=self.history.last_count,
                actual_anchor_frames=prepared['history_action_counts'], padded_frames=0,
                service_admission=response['history_admission'], privileged_truth_sent=False,
                evaluator_subgoal_sent=False, static_format_forced=True))
            # C1 expects a Sequence of float32 vectors, not a 2D ndarray.
            return {**response, 'actions': [row.copy() for row in actions]}

    return NativeTaskARPrefixClient(*args, **kwargs)

"""Official websocket transport, isolated from the physical-ACK diagnostic API.

Only one environment per connection: the official observation has no stable
environment/episode IDs with which to track a shrinking vector batch safely.
Different connections and resets never share stage, RNG, or inpainting state.
"""
import asyncio
import hashlib
import http
import time
import uuid

import numpy as np


CHUNK_KEY = '__action_chunk_size__'
CAMERAS = {
    'head_rgb': 'robot_r1::robot_r1:zed_link:Camera:0::rgb',
    'left_wrist_rgb': 'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb',
    'right_wrist_rgb': 'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb',
}
PROPRIO = 'robot_r1::proprio'


def parse_observation(request):
    keys = {'task_id', PROPRIO, *CAMERAS.values()}
    if not isinstance(request, dict) or set(request)-{CHUNK_KEY} != keys:
        raise ValueError('Only the pinned RGB/proprio/task observation is accepted')
    tasks = np.asarray(request['task_id'])
    if (tasks.dtype.kind not in 'iu' or tasks.shape not in ((1,), (1,1))
            or not 0 <= int(tasks.reshape(-1)[0]) < 100):
        raise ValueError('One environment and explicit valid task ID required')
    task = int(tasks.reshape(-1)[0])
    state = np.asarray(request[PROPRIO])
    if state.shape != (1,61) or state.dtype.kind != 'f' or not np.isfinite(state).all():
        raise ValueError('One finite R1Pro61 state required')
    images = {}
    for name,key in CAMERAS.items():
        pixels = np.asarray(request[key])
        if (pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[0] != 1
                or min(pixels.shape[1:3]) < 1 or pixels.shape[-1] not in (3,4)):
            raise ValueError('Unmodified single-environment NHWC uint8 RGB(A) required')
        images[name] = np.ascontiguousarray(pixels[0,...,:3].transpose(2,0,1))
    width = request.get(CHUNK_KEY, 0)
    if type(width) is not int or width not in (0,1,20):
        raise ValueError('Use default single-control requests or exact chunk20')
    return task, dict(images=images,proprio=np.array(state[0],dtype=np.float32,copy=True)), max(1,width)


class OfficialConnection:
    def __init__(self, session_factory, *, connection_id=None):
        self.session_factory = session_factory
        self.connection_id = connection_id or uuid.uuid4().hex
        self.epoch = 0
        self.session = self.prepared = self.width = None

    def reset(self):
        previous = None if self.session is None else self.session.finish()
        self.session = self.prepared = self.width = None
        self.epoch += 1
        return previous

    def prepare(self, request):
        if self.prepared is not None:
            raise ValueError('Prior response has not been delivered')
        task,observation,width = parse_observation(request)
        if self.session is None:
            episode = f'{self.connection_id}/reset-{self.epoch}/task-{task}'
            self.session = self.session_factory(episode,task)
            if self.session.clock_kind != 'policy_delivery':
                raise ValueError('Official transport must use a nonphysical delivery clock')
            self.session.begin()
            self.width = width
        if task != self.session.task:
            raise ValueError('Task changed without reset; never reuse another task history')
        if width != self.width:
            raise ValueError('Chunk negotiation cannot change within an episode')
        generated = None
        if self.session.pending is None:
            generated = self.session.infer(observation)
        actions = self.session.pending[:width].copy()
        if actions.shape != (width,23):
            raise ValueError('Chunk response cannot cross observation-conditioned boundaries')
        response = dict(action=actions[None,0,:].copy())
        if width == 20:
            response['action_chunk'] = actions[None].copy()
        self.prepared = dict(token=self.session.token,actions=actions,generated=generated)
        return response

    def delivered(self):
        if self.prepared is None:
            raise ValueError('No response to mark delivered')
        item = self.prepared
        result = self.session.record_delivery(token=item['token'],actions=item['actions'])
        self.prepared = None
        return dict(episode=self.session.episode,task=self.session.task,**result,
                    sent_controls=len(item['actions']),
                    action_sha256=hashlib.sha256(item['actions'].tobytes()).hexdigest(),
                    prediction=self.session.predictions,
                    generated_display=None if item['generated'] is None else item['generated']['display'])


class OfficialPolicyServer:
    """Serialized stateless model calls; websocket histories are per connection.

    This is protocol engineering, not a promise of concurrent GPU throughput or
    a substitute for a measured 24GB submission/container acceptance test.
    """
    def __init__(self, factory, *, metadata, packb, unpackb, journal=None, max_connections=8):
        if type(max_connections) is not int or max_connections < 1:
            raise ValueError('Positive connection limit required')
        self.factory,self.metadata = factory,dict(metadata)
        self.packb,self.unpackb = packb,unpackb
        self.journal = journal or (lambda row: None)
        self.max_connections = max_connections
        self.connections = set()
        self.model_lock = asyncio.Lock()

    @staticmethod
    async def health(connection, request):
        # websockets>=14 server API; legacy10 is supported for CPU test hosts.
        path = request.path if hasattr(request,'path') else connection
        if path == '/healthz':
            if hasattr(connection,'respond'):
                return connection.respond(http.HTTPStatus.OK,'OK\n')
            return http.HTTPStatus.OK,[('Content-Type','text/plain')],b'OK\n'
        return None

    async def handler(self, socket):
        if len(self.connections) >= self.max_connections:
            await socket.close(code=1013,reason='Registered connection capacity reached')
            return
        state = OfficialConnection(self.factory)
        self.connections.add(state.connection_id)
        self.journal(dict(kind='connected',connection=state.connection_id))
        try:
            await socket.send(self.packb(self.metadata))
            async for payload in socket:
                if not isinstance(payload,bytes):
                    raise ValueError('Binary msgpack requests required')
                request = self.unpackb(payload)
                if isinstance(request,dict) and 'reset' in request:
                    if set(request) != {'reset'} or request['reset'] is not True:
                        raise ValueError('Malformed reset')
                    self.journal(dict(kind='reset',connection=state.connection_id,previous=state.reset()))
                    # Official client reset() sends only: never enqueue a reply.
                    continue
                started = time.monotonic()
                async with self.model_lock:
                    response = await asyncio.to_thread(state.prepare,request)
                response['server_timing'] = dict(infer_ms=1000*(time.monotonic()-started))
                await socket.send(self.packb(response))
                self.journal(dict(kind='delivered',connection=state.connection_id,**state.delivered()))
        except Exception as error:
            # No automatic reconnect/replay: duplicate action delivery is unsafe.
            self.journal(dict(kind='connection_error',connection=state.connection_id,error=repr(error)))
            await socket.close(code=1011,reason='Invalid request or inference failure; consult server log')
        finally:
            self.journal(dict(kind='closed',connection=state.connection_id,previous=state.reset()))
            self.connections.discard(state.connection_id)

    async def run(self, *, host, port, stop=None, ready=None):
        import websockets
        async with websockets.serve(self.handler,host,port,compression=None,max_size=64<<20,
                                    ping_timeout=None,process_request=self.health) as server:
            if ready is not None:
                ready(server)
            if stop is None:
                await asyncio.Future()
            else:
                await stop.wait()

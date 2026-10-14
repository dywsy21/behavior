import asyncio
import ast
from copy import deepcopy
import functools
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time
import unittest
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Dict, Optional, Tuple
from unittest.mock import patch

import numpy as np
import websockets

from behavior_champion.official import (OfficialConnection,OfficialPolicyServer,
                                      parse_observation,CAMERAS,PROPRIO,CHUNK_KEY)
from behavior_champion.session import ChampionSession

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/eval/memlite_sft100'))
from wire import packb,unpackb


def observation(task=0, width=0):
    state=np.zeros((1,61),np.float32);state[:,24:26]=state[:,49:51]=.025
    result={key:np.zeros((1,4,5,4),np.uint8) for key in CAMERAS.values()}
    result.update({PROPRIO:state,'task_id':np.array([[task]],np.int64)})
    if width:result[CHUNK_KEY]=width
    return result


def predictor(obs,task,stage,previous,rng):
    noise=np.random.default_rng(int(rng[1])).uniform(0,.1,(30,23)).astype(np.float32)
    noise[:,0]+=task;noise[:,14]=noise[:,22]=.5
    if previous is not None:noise[:4]=previous
    logits=np.full(15,-np.inf,np.float32);logits[:5]=-2;logits[min(stage+1,4)]=2
    return dict(actions=noise,subtask_logits=logits),np.array([0,int(rng[1])+1],np.uint32)


def factory(episode,task,clock_kind='policy_delivery'):
    return ChampionSession(episode=episode,task=task,model_task=task,num_stages=5,seed=17,
        predictor=predictor,correct_actions=lambda t,s,p,a:(a,s),
        gripper_variation=lambda a,n:(False,0.,0.),apply_legacy_rules=False,clock_kind=clock_kind)


class OfficialContractTests(unittest.TestCase):
    def test_fresh_simulator_connections_equal_reset_stage_history_inpainting_and_rng(self):
        old=OfficialConnection(factory)
        for _ in range(11):old.prepare(observation(39,20));old.delivered()
        old.reset()
        fresh=OfficialConnection(factory)
        for _ in range(11):
            a=old.prepare(observation(39,20));old.delivered()
            b=fresh.prepare(observation(39,20));fresh.delivered()
            np.testing.assert_array_equal(a['action_chunk'],b['action_chunk'])
            np.testing.assert_array_equal(old.session.rng,fresh.session.rng)
            np.testing.assert_array_equal(old.session.initial_actions,fresh.session.initial_actions)
            self.assertEqual(old.session.stage,fresh.session.stage)
            self.assertEqual(list(old.session.history),list(fresh.session.history))
        self.assertNotEqual(old.session.episode,fresh.session.episode)

    def test_chunk_single_control_and_real_ack_policy_are_identical(self):
        chunks=OfficialConnection(factory);single=OfficialConnection(factory)
        physical=factory('physical',0,'physical_ack');physical.begin()
        for _ in range(5):
            expected=physical.infer(parse_observation(observation())[1])
            a=chunks.prepare(observation(width=20));chunks.delivered()
            np.testing.assert_array_equal(a['action'],a['action_chunk'][:,0])
            np.testing.assert_array_equal(a['action_chunk'][0],expected['action_chunk'])
            pieces=[]
            for _ in range(20):
                pieces.append(single.prepare(observation())['action'][0]);single.delivered()
            np.testing.assert_array_equal(np.stack(pieces),expected['action_chunk'])
            physical.ack(token=expected['token'],controls=[dict(control_step=physical.control_step+i,
                simulator_apply_ack=True,action_executed_raw23=x.tolist()) for i,x in enumerate(expected['action_chunk'])])
        self.assertEqual(single.session.control_step,0)
        self.assertEqual(single.session.clock()['physically_confirmed_controls'],None)
        self.assertEqual(single.session.delivered_controls,100)

    def test_interleave_and_reset_keep_all_state_private_including_partial_chunk(self):
        a=OfficialConnection(factory);b=OfficialConnection(factory);alone=OfficialConnection(factory)
        for _ in range(6):
            b.prepare(observation(1,20));b.delivered()
            x=a.prepare(observation(0,20));a.delivered()
            y=alone.prepare(observation(0,20));alone.delivered()
            np.testing.assert_array_equal(x['action_chunk'],y['action_chunk'])
        c=OfficialConnection(factory)
        first=c.prepare(observation())['action'];c.delivered()
        self.assertEqual(c.reset()['unused_offered_controls'],19)
        self.assertEqual(c.prepare(observation())['action'].tolist(),first.tolist())
        self.assertEqual(a.session.stage,alone.session.stage)
        self.assertEqual(list(a.session.history),list(alone.session.history))
        np.testing.assert_array_equal(a.session.rng,alone.session.rng)
        np.testing.assert_array_equal(a.session.initial_actions,alone.session.initial_actions)
        c.reset();c.prepare(observation(1));self.assertEqual(c.session.task,1)

    def test_no_fabricated_ack_and_no_delivery_before_send_or_wrong_bytes(self):
        c=OfficialConnection(factory);response=c.prepare(observation(width=20))
        self.assertEqual(c.session.delivered_controls,0)
        with self.assertRaises(ValueError):c.prepare(observation(width=20))
        with self.assertRaises(ValueError):c.session.ack(token=c.session.token,controls=[{}])
        wrong=response['action_chunk'][0].copy();wrong[0,0]+=1
        with self.assertRaises(ValueError):c.session.record_delivery(token=c.session.token,actions=wrong)
        receipt=c.delivered();self.assertIsNone(receipt['physically_confirmed_controls'])
        self.assertNotIn('control_step',receipt)
        with self.assertRaises(ValueError):c.delivered()
        with self.assertRaises(ValueError):factory('p',0,'physical_ack').record_delivery(token='',actions=wrong)

    def test_foreign_task_vector_batch_privileged_fields_and_changed_chunk_rejected(self):
        bad=observation();bad['physical_success']=True
        with self.assertRaises(ValueError):parse_observation(bad)
        for shape in ((2,),(),(2,1)):
            bad=observation();bad['task_id']=np.zeros(shape,np.int64)
            with self.assertRaises(ValueError):parse_observation(bad)
        for width in (16,32,True,20.0):
            bad=observation();bad[CHUNK_KEY]=width
            with self.assertRaises(ValueError):parse_observation(bad)
        c=OfficialConnection(factory);c.prepare(observation());c.delivered()
        with self.assertRaises(ValueError):c.prepare(observation(1))
        with self.assertRaises(ValueError):c.prepare(observation(0,20))


class OfficialSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Test local sockets directly; a user's HTTP proxy is not the server.
        self.proxy_guard=patch.dict(os.environ,{'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'})
        self.proxy_guard.start();self.addCleanup(self.proxy_guard.stop)
        self.stop=asyncio.Event();ready=asyncio.get_running_loop().create_future()
        self.rows=[]
        self.server=OfficialPolicyServer(factory,metadata={'action_horizon':20},packb=packb,unpackb=unpackb,
            journal=self.rows.append)
        self.job=asyncio.create_task(self.server.run(host='127.0.0.1',port=0,stop=self.stop,
            ready=lambda s:ready.set_result(s.sockets[0].getsockname()[1])))
        self.port=await asyncio.wait_for(ready,10)

    async def asyncTearDown(self):
        self.stop.set();await asyncio.wait_for(self.job,10)

    async def test_health_metadata_no_reset_reply_and_two_independent_connections(self):
        def health():
            with urlopen(f'http://127.0.0.1:{self.port}/healthz',timeout=5) as r:return r.read()
        self.assertEqual(await asyncio.to_thread(health),b'OK\n')
        async with websockets.connect(f'ws://127.0.0.1:{self.port}') as a, \
                   websockets.connect(f'ws://127.0.0.1:{self.port}') as b:
            self.assertEqual(unpackb(await a.recv()),{'action_horizon':20});await b.recv()
            await a.send(packb({'reset':True}))
            with self.assertRaises(asyncio.TimeoutError):await asyncio.wait_for(a.recv(),.04)
            await a.send(packb(observation(0,20)));first=unpackb(await a.recv())
            await b.send(packb(observation(1,20)));other=unpackb(await b.recv())
            self.assertFalse(np.array_equal(first['action'],other['action']))
            await a.send(packb({'reset':True}));await a.send(packb(observation(0,20)))
            restarted=unpackb(await a.recv())
            np.testing.assert_array_equal(first['action_chunk'],restarted['action_chunk'])
            np.testing.assert_array_equal(first['action'],first['action_chunk'][:,0])
        self.assertTrue(all(row['physically_confirmed_controls'] is None
                            for row in self.rows if row['kind']=='delivered'))

    @unittest.skipUnless(importlib.util.find_spec('websockets.sync'), 'Exact official sync client needs websockets>=14')
    async def test_exact_pinned_official_client_health_reset_and_chunk_consumption(self):
        # Load only original network code, not OmniGibson/Isaac. No patched act/reset.
        source=Path(os.environ.get('CHAMPION_OFFICIAL_NETWORK_SOURCE',str(REPO/
            'artifacts/rl-recovery-10383-20261008/official-behavior/OmniGibson/omnigibson/eval/utils/network_utils.py')))
        if not source.exists():self.skipTest('Pinned official source not available on this test host')
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         'a9cc50bd502e1a4599db353f7afbd22d371bd5aa0ba22b6dfc412b09de4e91d1')
        import requests,torch as th,msgpack
        import websockets.sync.client
        namespace=dict(asyncio=asyncio,functools=functools,np=np,th=th,msgpack=msgpack,requests=requests,
            websockets=websockets,logger=logging.getLogger('official-client'),time=time,
            urlparse=urlparse,deepcopy=deepcopy,Dict=Dict,Optional=Optional,Tuple=Tuple,
            ACTION_CHUNK_REQUEST_KEY=CHUNK_KEY)
        nodes=[node for node in ast.parse(source.read_text()).body if
               isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in
               ('WebsocketClientPolicy','pack_data','unpack_data')]
        # Serializer aliases in the original module, used by its unmodified class.
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        namespace['Packer']=functools.partial(msgpack.Packer,default=namespace['pack_data'])
        namespace['unpackb']=functools.partial(msgpack.unpackb,object_hook=namespace['unpack_data'])
        def exercise():
            client=namespace['WebsocketClientPolicy'](host='127.0.0.1',port=self.port,action_chunk_size=20)
            try:
                client.reset();actions=[client.act(observation()) for _ in range(41)]
                client.reset();reset=client.act(observation())
                np.testing.assert_array_equal(actions[0].numpy(),reset.numpy())
                self.assertTrue(all(x.shape==(1,23) for x in actions))
            finally:
                if client._ws is not None:client._ws.close()
        await asyncio.wait_for(asyncio.to_thread(exercise),20)
        delivered=[row for row in self.rows if row['kind']=='delivered']
        self.assertEqual(len(delivered),4)  # Three chunks for41 controls, one after reset.
        self.assertEqual([row['delivered_controls'] for row in delivered],[20,40,60,20])


if __name__=='__main__':unittest.main()

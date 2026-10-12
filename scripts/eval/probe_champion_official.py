"""Finite actual-model protocol check using the original official client class.

Replays two TRAIN observations only. The returned controls are NOT simulated.
"""
import argparse
import ast
from copy import deepcopy
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import msgpack
import numpy as np
import requests
import torch as th
import websockets
import websockets.sync.client

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/eval/memlite_sft100')]
from behavior_champion.inference import file_sha
from behavior_champion.official import CAMERAS,PROPRIO,CHUNK_KEY
from common import atomic_json


def official_client(path):
    if file_sha(path)!='a9cc50bd502e1a4599db353f7afbd22d371bd5aa0ba22b6dfc412b09de4e91d1':
        raise ValueError('Unmodified pinned official network source required')
    namespace=dict(functools=functools,np=np,th=th,msgpack=msgpack,requests=requests,websockets=websockets,
        logger=logging.getLogger('official-client'),time=time,urlparse=urlparse,deepcopy=deepcopy,
        Dict=Dict,Optional=Optional,Tuple=Tuple,ACTION_CHUNK_REQUEST_KEY=CHUNK_KEY)
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef))
           and node.name in ('WebsocketClientPolicy','pack_data','unpack_data')]
    if len(nodes)!=3:raise ValueError('Incomplete official client module')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    namespace['Packer']=functools.partial(msgpack.Packer,default=namespace['pack_data'])
    namespace['unpackb']=functools.partial(msgpack.unpackb,object_hook=namespace['unpack_data'])
    return namespace['WebsocketClientPolicy']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--official-network-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.output.exists():raise FileExistsError(args.output)
    source=args.official_network_source
    Client=official_client(source)
    meta=json.loads((args.probe/'result.json').read_text())
    if file_sha(args.probe/'native.npz')!=meta['native_npz_sha256']:
        raise ValueError('Changed genuine TRAIN probe')
    if cfg['allowed_task_ids']!=[0,1] or cfg['host']!='127.0.0.1':
        raise ValueError('Registered two-task loopback engineering probe required')
    native=np.load(args.probe/'native.npz',allow_pickle=False)
    observations={}
    for task in (0,1):
        indices=np.flatnonzero(native['task_id']==task)
        if len(indices)!=1:raise ValueError('Exactly one source per registered task required')
        i=int(indices[0])
        obs={'task_id':np.array([[task]],np.int64),PROPRIO:native['state61'][i:i+1]}
        for camera,key in [('head_rgb','base_0_rgb'),('left_wrist_rgb','left_wrist_0_rgb'),
                           ('right_wrist_rgb','right_wrist_0_rgb')]:
            obs[CAMERAS[camera]]=native[key][i:i+1]
        observations[task]=obs
    # Only this process: loopback must never be sent through a configured proxy.
    for key in ('NO_PROXY','no_proxy'):
        os.environ[key]=','.join(filter(None,[os.environ.get(key,''),'127.0.0.1','localhost']))
    args.output.mkdir(parents=True)
    status=dict(status='running',started=time.time(),config_sha256=file_sha(args.config),
        official_network_sha256=file_sha(source),native_probe_sha256=meta['native_npz_sha256'],
        optimizer_updates=0,physical_controls=0,controls_returned_are_not_simulated=True,checks=[])
    atomic_json(args.output/'status.json',status)
    clients=[]
    def new(width):
        client=Client(host=cfg['host'],port=cfg['port'],action_chunk_size=width,allow_reconnect=False)
        clients.append(client);client.reset()
        metadata=client.get_server_metadata()
        if (metadata['config_sha256']!=status['config_sha256'] or metadata['environment_batch_size']!=1
                or metadata['physical_execution_evidence'] is not False):
            raise ValueError('Wrong server identity or clock')
        return client
    def close(client):
        if client._ws is not None:client._ws.close()
        clients.remove(client)
    def controls(client,task,n):
        result=np.stack([client.act(observations[task]).numpy()[0] for _ in range(n)])
        if result.shape!=(n,23) or not np.isfinite(result).all():raise ValueError('Invalid native controls')
        return result
    def check(label,actual,expected):
        np.testing.assert_array_equal(actual,expected)
        status['checks'].append(dict(name=label,controls=len(actual),exact=True,
            actions_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        atomic_json(args.output/'status.json',status)
    try:
        expected={}
        for task in (0,1):
            client=new(20);expected[task]=controls(client,task,60);close(client)
        paired={task:new(20) for task in (0,1)}
        for turn in range(3):
            for task in (1,0):
                check(f'interleaved-task{task}-chunk{turn}',controls(paired[task],task,20),
                      expected[task][20*turn:20*(turn+1)])
        for client in paired.values():close(client)
        single=new(0)
        check('single-control-vs-chunk20-task0',controls(single,0,60),expected[0]);close(single)
        reset=new(20)
        check('partial-chunk-first-control',controls(reset,0,1),expected[0][:1])
        reset.reset()
        check('reset-drops-nineteen-cached-controls-and-all-history',controls(reset,0,20),expected[0][:20])
        reset.reset()
        check('reset-may-change-task-with-no-state-leak',controls(reset,1,20),expected[1][:20]);close(reset)
        status.update(status='passed_actual_model_official_protocol',finished=time.time(),
                      checks_passed=len(status['checks']),not_physical_effectiveness=True)
        atomic_json(args.output/'result.json',status)
    except BaseException as error:
        status.update(status='failed_no_retry',error=repr(error),finished=time.time())
        raise
    finally:
        for client in list(clients):close(client)
        atomic_json(args.output/'status.json',status)
    print(json.dumps(status,allow_nan=False))


if __name__=='__main__':main()

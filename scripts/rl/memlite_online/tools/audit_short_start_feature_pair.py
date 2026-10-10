"""Compare real short-attempt features to the immutable cadence16 control."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_corpus import digest, file_sha


def compare(old, new):
    import torch
    if old['history_protocol'] != 'cadence16_v1' or new['history_protocol'] != 'attempt_start_short_v1':
        raise ValueError('Explicit old/new history protocols required')
    old_requests = {r['request_id']: r for r in old['requests']}
    new_requests = {r['request_id']: r for r in new['requests']}
    if (len(old_requests) != len(old['requests']) or len(new_requests) != len(new['requests'])
            or set(old_requests) != set(new_requests)
            or set(old['features']) != set(old_requests) or set(new['features']) != set(new_requests)):
        raise ValueError('Short history cannot change any selected source/label/member request')
    if set(old['initial_head']) != set(new['initial_head']) or any(
            not torch.equal(v, new['initial_head'][k]) for k, v in old['initial_head'].items()):
        raise ValueError('Initial result head changed')
    unchanged = extended = 0
    for key, before in old_requests.items():
        after = new_requests[key]
        metadata = lambda r: {k: v for k, v in r.items() if k not in ('checks', 'history_protocol')}
        if metadata(before) != metadata(after):
            raise ValueError('Request identity or causal role changed')
        for cache,request in ((old,before),(new,after)):
            features=cache['features'][key]
            if (features['steps'].tolist()!=[c['control_step'] for c in request['checks']]
                    or any(len(features[n])!=len(request['checks']) or not torch.isfinite(features[n]).all()
                           for n in ('context','proprio'))):
                raise ValueError('Feature times, shape or finiteness differ from real observations')
        if after['checks'] == before['checks']:
            unchanged += 1
            offset = 0
        else:
            a, b = before['checks'], after['checks']
            if (len(a) != 1 or len(b) != 2 or a[0] != b[1]
                    or b[0]['served_controls'] != 0 or not 0 < b[1]['served_controls'] < 16
                    or b[1]['control_step'] - b[0]['control_step'] != b[1]['served_controls']
                    or b[0]['sample_id'] == b[1]['sample_id']
                    or any(b[0][k] != b[1][k] for k in ('task_name','parent_goal','issued_bundle','member_index'))):
                raise ValueError('Only the actual distinct same-attempt issuance frame may be prepended')
            extended += 1
            offset = 1
        for name in ('context', 'proprio', 'steps'):
            if not torch.equal(old['features'][key][name], new['features'][key][name][offset:]):
                raise ValueError('Previously consumed current/history features changed: ' + name)
    return dict(status='short_start_pair_passed',requests=len(old_requests),unchanged=unchanged,
        extended_same_attempt_starts=extended,all_existing_features_bitwise_equal=True,
        initial_head_bitwise_equal=True,optimizer_updates=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('control','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    receipts=[];caches=[]
    for path in (a.control,a.candidate):
        r=json.loads((path/'receipt.json').read_text())
        if r['diagnostic_only'] or r['optimizer_steps'] or file_sha(path/'features.pt')!=r['features_sha256']:
            raise ValueError('Real immutable admitted feature cache required')
        c=torch.load(path/'features.pt',map_location='cpu',weights_only=False)
        if digest(c['requests'])!=r['requests_sha256'] or c['history_protocol']!=r['history_protocol']:
            raise ValueError('Feature request receipt changed')
        receipts.append(r);caches.append(c)
    for key in ('high_sha256','admission_sha256','inventory_sha256','history_sha256','stats_sha256'):
        if receipts[0][key]!=receipts[1][key]:raise ValueError('Changed parent/data/normalization: '+key)
    result=compare(*caches)
    result.update(control_features_sha256=receipts[0]['features_sha256'],
        candidate_features_sha256=receipts[1]['features_sha256'],admission_sha256=receipts[0]['admission_sha256'])
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

"""Matched, NON-ACTUATING input sensitivity test on one frozen failed decision.

Nine calls, interleaved three ways, no response retries or selected subset.
Nothing here changes the active multi-task actor or issues motor commands.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from semantic_robot.v2.jev_client import JevClient, MODEL

SOURCE_SHA='eefd27fccc3cb868aa351404eb164e3701853116e9a1085521a1e2be90b33aa5'
ORDER=('exact','search_explanation','rotation_label')
EXPLANATION=(
    '\nStage-specific interpretation for WORLD SEARCH of an unseen target: '
    'target distance and contact progress are unavailable until the target is found. '
    'They are not the objective of a search command. A currently preflighted viewpoint change may be chosen '
    'to gather information without promising target visibility. Heading coverage is quantized into 24 bins; '
    'an empty predicted_new_heading_bins list does not prove that a small yaw leaves the image unchanged. '
    'These are interface semantics, not new observations or a guarantee of safe unobserved space. '
    'Retain every actual safety veto, command limit, carrying constraint and recorded failure. '
    'Choose the command yourself; HOLD and abstain remain available. No preferred direction is prescribed.'
)


def variants(raw):
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA: raise ValueError('Only the frozen actual JEV-04 decision3')
    request=json.loads(raw)['request_without_pixel_duplicates']
    state=request['state']
    if (request['model']!=MODEL or request['images'] or set(request['questions'])!={'command'}
            or state['harness']['stage']!='SEARCH' or state['facts']['target_visible'] is not False
            or state['chosen_tactic_not_new_evidence']!='search' or state['harness']['search']['total_bins']!=24):
        raise ValueError('Wrong observed context')
    result={name:copy.deepcopy(request) for name in ORDER}
    # B changes ONLY a question instruction, not sensor facts or allowed options.
    result['search_explanation']['questions']['command']['instructions']+=EXPLANATION
    # C tests ONLY the ambiguous generic rotation label. Its replacement is
    # computed from actual offered yaw/roll/pitch commands; no geometry is invented.
    available=any(v['command']['move'].startswith(('yaw_','roll_','pitch_'))
                  for v in state['commands'].values())
    if not available or state['facts']['rotation_option_available'] is not False:
        raise ValueError('Expected the recorded base-yaw / false generic-rotation mismatch')
    result['rotation_label']['state']['facts']['rotation_option_available']=available
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--key-file',required=True)
    args=parser.parse_args()
    if 'TYPESAFE_API_KEY' in os.environ:raise ValueError('Private key file only')
    inputs=variants(Path(args.source).read_bytes())
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Use clean frozen probe source')
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    def write(name,value):(folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    result=dict(schema='jev-search-input-sensitivity-v1',source_sha256=SOURCE_SHA,
        code_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        model=MODEL,request_bound=9,wall_seconds=240,new_controls=0,new_resets=0,training_updates=0,
        active_multitask_actor_changed=False,not_a_success_rate_evaluation=True,rows=[],completed=False)
    write('inputs.json',inputs)
    api=None
    with (folder/'calls.jsonl').open('x') as stream:
        def journal(row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        try:
            api=JevClient(key_file=args.key_file,max_calls=9,journal=journal)
            api.deadline=time.perf_counter()+240
            for repeat in range(3):
                # Rotate order to avoid testing all baselines first.
                for name in ORDER[repeat:]+ORDER[:repeat]:
                    request=inputs[name]
                    answer,receipt=api.evaluate(request['state'],request['questions'])
                    value=answer['answers']['command']; selected=value['choice']
                    row=dict(repeat=repeat,variant=name,choice=selected,probabilities=value['probabilities'],
                        action=None if selected=='abstain' else request['state']['commands'][selected]['command'])
                    result['rows'].append(row);write(f'choice_{len(result["rows"]):02d}.json',receipt)
                    print(json.dumps(row),flush=True)
            result['completed']=True
        except BaseException as error:
            result['error']=str(error);raise
        finally:
            if api:result.update(api_requests=api.calls,validated_responses=api.validated_responses,
                                 input_tokens=api.input_tokens,output_tokens=api.output_tokens)
            write('result.json',result)


if __name__=='__main__':main()

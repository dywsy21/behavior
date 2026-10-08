import ast
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from resume_plan import remaining_parts
from publication import evaluation_commands,command_task_names
from common import atomic_json


class ResumeTests(unittest.TestCase):
    def test_queue_claims_exclusive_and_never_retries(self):
        tree=ast.parse((SOURCE/'resume_worker.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='claim')
        scope=dict(fcntl=fcntl,atomic_json=atomic_json,os=os,time=time)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'claim-production','exec'),scope)
        claim=scope['claim']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'claims').mkdir()
            parts=[dict(id=f'part{i}') for i in range(64)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows=list(pool.map(lambda i:claim(root,parts,i%8),range(80)))
            ids=[row['id'] for row in rows if row is not None]
            self.assertEqual(len(ids),64)
            self.assertEqual(len(set(ids)),64)
            self.assertIsNone(claim(root,parts,0))
            (root/'STOP').touch()
            with self.assertRaises(InterruptedError):claim(root,parts,0)

    def test_every_remaining_case_once_and_no_duplicate_tail_padding(self):
        tasks=[f'task{i}' for i in range(100)];h={task:i+10 for i,task in enumerate(tasks)}
        done=[dict(task=task,instance_id=i,rollout_id=0) for task in tasks[:8] for i in range(301,309)]
        for n in (2,4):
            parts=remaining_parts(tasks,done,h,n)
            cases=[(part['task'],i+301) for part in parts for i in part['indices']]
            self.assertEqual(len(cases),936)
            self.assertEqual(len(set(cases)),936)
            self.assertFalse(set(cases)&{(d['task'],d['instance_id']) for d in done})
            self.assertTrue(all(len(p['indices'])%p['num_envs']==0 for p in parts))
            weights=[p['scheduling_weight'] for p in parts]
            self.assertEqual(weights,sorted(weights,reverse=True))

    def test_one_completed_row_in_interrupted_batch_does_not_replay(self):
        done=[dict(task='task',instance_id=301,rollout_id=0)]
        parts=remaining_parts(['task'],done,dict(task=20),4)
        self.assertEqual(sorted(p['num_envs'] for p in parts),[1,4])
        self.assertEqual({i for p in parts for i in p['indices']},set(range(1,10)))

    def test_bad_or_duplicate_inherited_case_rejected(self):
        record=dict(task='task',instance_id=301,rollout_id=0)
        for rows in [[record,record],[record|dict(instance_id=1)],[record|dict(rollout_id=1)]]:
            with self.assertRaises(ValueError):remaining_parts(['task'],rows,dict(task=20),4)

    def test_full_coverage_needs_no_new_work(self):
        done=[dict(task='task',instance_id=i,rollout_id=0) for i in range(301,311)]
        self.assertEqual(remaining_parts(['task'],done,dict(task=20),4),[])

    def test_command_provenance_preserves_prior_and_new_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory);part=job/'parts/task_b_n4';part.mkdir(parents=True)
            (part/'command.json').write_text(json.dumps(dict(part=dict(task='task_b'),argv=['new'])))
            (job/'prior_commands.json').write_text(json.dumps([
                dict(path='/prior/tasks/task_a/command.json',command=dict(argv=['original']))]))
            manifest=dict(kind='native_sft100_admin_resume')
            commands=evaluation_commands(job,manifest)
            self.assertEqual(command_task_names(commands,manifest),{'task_a','task_b'})
            self.assertEqual(commands[0]['command']['argv'],['original'])


if __name__=='__main__':unittest.main()

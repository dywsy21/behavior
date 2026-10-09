from copy import deepcopy
import json
from pathlib import Path
import sys
import pickle
import tempfile
import unittest
import zipfile
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import CAMERAS,file_sha
from recovery_terminal_corpus import load_terminal,normalized_terminal,terminal_anchor
from recovery_sft_data import CandidateArchiveReader


class TerminalCorpusTests(unittest.TestCase):
    def build(self, root):
        physical=dict(target_name='cup',entity='cup.n.01_1',grasp={'left':'FALSE','right':'FALSE'},
                      gripper_aperture={'left':.001,'right':.05})
        context=dict(control_step=0,active_skills_semantic_json='[{"verb":"GRASP","target":"cup"}]',parent_goal='goal')
        rows=[dict(control_step=i,proprio_after=[float(i+1)]*61,physical_audit=deepcopy(physical),
                   teacher=dict(stage='CLOSE',joint_error_max_rad=.001)) for i in range(6)]
        terminal=dict(schema='recovery_terminal_observation_v1',control_step=6,last_applied_control_step=5,
            proprio=[6.]*61,physical_audit=physical,context=context,has_next_executed_action=False,
            training_approved=False,outcome_candidate='FAILED',
            reason='Closed empty grasp at settled reference pose; correction failed',images={})
        (root/'terminal_rgb').mkdir()
        for camera in CAMERAS:
            path=root/'terminal_rgb'/(camera+'.jpg');Image.new('RGB',(224,224),'white').save(path)
            terminal['images'][camera]=dict(path=str(path.relative_to(root)),sha256=file_sha(path))
        path=root/'terminal-observation.json';path.write_text(json.dumps(terminal))
        manifest=dict(arm='left',terminal_observation_sha256=file_sha(path),kind='open_gripper',
                      physical_recovery_candidate=False,failure=terminal['reason'])
        return terminal,rows,manifest,[context]

    def test_exact_final_evidence_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);t,rows,m,plans=self.build(root)
            self.assertEqual(load_terminal(root,m,rows,plans),t)
            broken=deepcopy(rows);broken[-1]['proprio_after'][0]=7.
            with self.assertRaisesRegex(ValueError,'another physical time'):load_terminal(root,m,broken,plans)
            with self.assertRaisesRegex(ValueError,'context identity'):load_terminal(root,m,rows,[dict(plans[0],parent_goal='other')])
            (root/t['images']['head_rgb']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'original final RGB'):load_terminal(root,m,rows,plans)

    def test_real_reader_accepts_image_only_tail_but_never_next_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);t,rows,m,plans=self.build(root)
            episode=dict(run='offline',episode_id='one',task='example_task',instance_id=1)
            image_ref=dict(archive='one.zip',control_step=6,sha256={k:v['sha256'] for k,v in t['images'].items()})
            row=terminal_anchor(episode,t,image_ref,'train',m)
            self.assertFalse(row['label_audit']['full_executed_32_step_target_available'])
            self.assertEqual(row['label_audit']['outcome']['value'],'FAILED')
            self.assertNotIn('physical_audit',row['actor_input'])
            self.assertNotIn('reason',row['actor_input'])
            header=dict(episode=episode,rgb_anchors={'6':dict(sha256=image_ref['sha256'])},
                        terminal_observation=normalized_terminal(t,{'cup':'cup.n.01_1'}))
            with zipfile.ZipFile(root/'one.zip','w') as z:
                z.writestr('manifest.json',json.dumps(header))
                z.writestr('transitions.jsonl',''.join(json.dumps(x)+'\n' for x in rows))
                for camera,ref in t['images'].items():z.write(root/ref['path'],f'rgb/00000006/{camera}.jpg')
            reader=CandidateArchiveReader(root,[dict(path='one.zip',episode=episode,sha256=file_sha(root/'one.zip'))])
            state,images=reader.observation(row)
            original_handle = reader.archives['one.zip'][1]
            again_state, again_images = reader.observation(row)
            self.assertIs(reader.archives['one.zip'][1], original_handle)
            self.assertEqual(state.tolist(), again_state.tolist())
            for camera in CAMERAS:
                self.assertEqual(images[camera].tobytes(), again_images[camera].tobytes())
            transferred = pickle.loads(pickle.dumps(reader))
            self.assertFalse(transferred.archives)
            self.assertEqual(transferred.observation(row)[0].tolist(),state.tolist())
            transferred.close()
            self.assertEqual(state.shape,(1,61));self.assertEqual(set(images),set(CAMERAS))
            self.assertEqual(set(reader.episode(['offline','one'])[0]),set(range(6)))
            with self.assertRaisesRegex(ValueError,'no next action'):reader.observation_and_actions(row)
            altered=deepcopy(row);altered['label_audit']['full_executed_32_step_target_available']=True
            with self.assertRaisesRegex(ValueError,'confused with an applied action'):reader.observation(altered)
            with zipfile.ZipFile(root/'one.zip','a') as changed:
                changed.writestr('changed.txt', 'immutable sources cannot change')
            with self.assertRaisesRegex(ValueError,'changed while open'):reader.observation(row)
            reader.close()


if __name__=='__main__':unittest.main()

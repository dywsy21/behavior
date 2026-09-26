from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/vlm_sft'))
import inspect_expert_action_corpus as review


class ActionReviewTests(unittest.TestCase):
    def test_source_root_and_alias_outputs_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset';root.mkdir()
            alias=Path(tmp)/'alias';alias.symlink_to(root,target_is_directory=True)
            with patch.object(review,'ROOT',root):
                for path in (root/'review_out',alias/'review_out'):
                    with self.assertRaises(ValueError):review.guard_output(path,Path(tmp)/'corpus',[])
                review.guard_output(Path(tmp)/'new_review',Path(tmp)/'corpus',[])

    def rows(self):
        rows=[]
        for f in range(4):
            a=np.zeros((16,23));a[:,0]=f*.1
            if f==2:a[8:,14]=-1
            rows.append({'id':f'row{f}','frame_index':f*16,'observation_state':[0.]*61,'expert_action':a.tolist()})
        return rows

    def test_source_sampling_is_train_only_and_fifty_unique_groups(self):
        manifest={'shards':[{'task':t,'instance':i,'episode':t*100+i,'samples':20,
            'split':'train' if i<12 else 'test'} for t in range(5) for i in range(16)]}
        selected=review.select_shards(manifest)
        self.assertEqual(len(selected),50)
        self.assertTrue(all(s['split']=='train' for s,_ in selected))
        manifest['shards'].reverse()
        self.assertEqual(selected,review.select_shards(manifest))

    def test_stratum_fallback_is_explicit_not_fabricated(self):
        rows=self.rows()
        row,fallback=review.choose_row(rows,'torso')
        self.assertTrue(fallback)
        self.assertFalse(review.command_activity(row)['torso'])
        row,fallback=review.choose_row(rows,'gripper')
        self.assertFalse(fallback)
        self.assertEqual(row['id'],'row2')

    def test_temporal_quantiles_and_bad_arrays(self):
        rows=self.rows()
        self.assertEqual(review.choose_row(rows,'early')[0]['frame_index'],0)
        self.assertEqual(review.choose_row(rows,'late')[0]['frame_index'],48)
        rows[0]['expert_action'][0][3]=float('nan')
        with self.assertRaises(ValueError):review.command_activity(rows[0])


if __name__=='__main__':unittest.main()

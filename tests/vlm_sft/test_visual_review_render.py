import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/vlm_sft'))
import render_visual_review as review


class ReviewRenderTests(unittest.TestCase):
    def test_raw_review_preserves_originals_and_is_never_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir();paths={};receipts={}
            for view in ('head','left_wrist','right_wrist'):
                path=root/'images'/f'row_{view}.png';Image.new('RGB',(480,480),(20,40,60)).save(path)
                paths[view]=str(path.relative_to(root));receipts[view]={'png_sha256':review.sha(path)}
            row={'id':'row','task':0,'instance':10,'frame':0,'split':'train','images':paths,'image_receipts':receipts}
            inventory={'review_states':[row],'manifest_sha256':'a'*64};out=root/'sheets'
            pages=review.raw_sheets(inventory,root,out);self.assertEqual(len(pages),1)
            self.assertFalse(json.loads((out/'sheets.json').read_text())['training_eligible'])
            for view,path in paths.items():self.assertEqual(review.sha(root/path),receipts[view]['png_sha256'])
            row['split']='test'
            with self.assertRaises(ValueError):review.raw_sheets(inventory,root,root/'forbidden')

    def test_teacher_overlay_does_not_modify_input_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir();path=root/'images/row.png'
            Image.new('RGB',(480,480),(20,40,60)).save(path);digest=review.sha(path)
            predictions=[{'id':'row','phase':'primary','png_sha256':digest,'query':'the radio','expected_visibility':'P',
                          'format_valid':True,'parsed':{'visibility':'present','boxes':[[0,0,1000,1000]]}}]
            pages=review.teacher_sheets(predictions,root,root/'sheets');self.assertEqual(len(pages),1)
            self.assertEqual(review.sha(path),digest)


if __name__=='__main__':unittest.main()

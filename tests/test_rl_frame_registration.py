from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semantic_robot.v2.render_batch import OwnedFrameRegistration,FRAME_ANNOTATOR


@dataclass
class Template:
    node_type_id: str='omni.syntheticdata.SdFrameIdentifier'
    attributes: dict=field(default_factory=dict)


class FrameRegistrationTests(unittest.TestCase):
    def test_repeat_owned_unchanged_registration_without_native_overwrite(self):
        synthetic=SimpleNamespace(_ogn_templates_registry={})
        def register(rep,s):
            if FRAME_ANNOTATOR in s._ogn_templates_registry: raise ValueError('already registered')
            s._ogn_templates_registry[FRAME_ANNOTATOR]=Template()
        token=OwnedFrameRegistration()
        with patch('semantic_robot.v2.render_batch.register_frame_annotator',side_effect=register) as call:
            token.ensure(None,synthetic); token.ensure(None,synthetic)
            self.assertEqual(call.call_count,1)
            synthetic._ogn_templates_registry[FRAME_ANNOTATOR].attributes['wrong']=1
            with self.assertRaises(ValueError): token.ensure(None,synthetic)
            with self.assertRaises(ValueError): OwnedFrameRegistration().ensure(None,synthetic)

    def test_identical_replacement_is_still_not_owned(self):
        s=SimpleNamespace(_ogn_templates_registry={}); token=OwnedFrameRegistration()
        with patch('semantic_robot.v2.render_batch.register_frame_annotator',
                   side_effect=lambda rep,synthetic:synthetic._ogn_templates_registry.update({FRAME_ANNOTATOR:Template()})):
            token.ensure(None,s)
        s._ogn_templates_registry[FRAME_ANNOTATOR]=Template()
        with self.assertRaises(ValueError): token.ensure(None,s)


if __name__=='__main__': unittest.main()

import importlib.util
import ast
from pathlib import Path
import unittest

from semantic_robot.v2.protocol import strict_json
from semantic_robot.v2.protocol import HOLD
from semantic_robot.v2.affordance import SurfaceChoice
from semantic_robot.v2.grounded_harness import parse_recovery

path = Path(__file__).resolve().parents[2]/"scripts/semantic_robot/serve_v2.py"
spec = importlib.util.spec_from_file_location("semantic_v2_service",path)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class TransportTests(unittest.TestCase):
    def test_actual_b1_recovery_text_is_audit_not_actuation(self):
        # Exact H-08 B1 calls77/82. Only rationale length failed previously.
        texts=[
            '{"strategy": "move_forward", "visible_reason": "A complete 360-degree sweep at this viewpoint found no target; the visible open floor and clear path toward the far glass doors suggest the breakfast table is in an adjacent area, so moving forward to a new viewpoint is the only supported option."}',
            '{"strategy":"hold","visible_reason":"The red and white radio is clearly visible on the table in front of the robot, but the recovery budget is exhausted and the target surface estimate is invalid due to multiview disagreement. Holding is the safest option to avoid further failed approaches."}'
        ]
        for text in texts:
            self.assertEqual(parse_recovery(text),strict_json(text))
        import json
        for value in ({"strategy":"teleport","visible_reason":"clear"},
                      {"strategy":"hold","visible_reason":"x"*1025},
                      {"strategy":"hold","visible_reason":"ok","done":True}):
            with self.assertRaises(ValueError):parse_recovery(json.dumps(value))

    def test_action_and_ground_grammars_remain_separate(self):
        service.validate_scoped_choices("act",[HOLD.text()])
        service.validate_scoped_choices("ground",[SurfaceChoice().text(),SurfaceChoice(2).text()])
        service.validate_scoped_choices("observe",[])
        for kind,lines in (("ground",[SurfaceChoice(2).text()]),("ground",[HOLD.text()]),
                           ("act",[SurfaceChoice().text()]),("ground",[SurfaceChoice().text()]*2),
                           ("ground",['{"candidate_id": null}']), ("ground",[None]),
                           ("act",[]),("plan",[SurfaceChoice().text()])):
            with self.subTest(kind=kind,lines=lines),self.assertRaises(ValueError):
                service.validate_scoped_choices(kind,lines)

    def test_portable_modules_parse_on_model_environment_python310(self):
        root=Path(__file__).resolve().parents[2]
        for path in (root/"src/semantic_robot/v2").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(),feature_version=(3,10))

    def test_plain_unchanged_and_one_fence_only(self):
        payload = '{"visible":true,"note":"observed"}'
        self.assertEqual(service.normalize_json_transport(payload),(payload,None))
        for tag in ("```json","```"):
            value,wrapper = service.normalize_json_transport(tag+"\n"+payload+"\n```")
            self.assertEqual(value,payload); self.assertIsNotNone(wrapper)

    def test_not_a_general_json_repair_or_prose_extractor(self):
        invalid = ['Here: ```json\n{}\n```', '```json\n{}', '```json\n{}\n```\nExtra',
                   '```json\n{"x":1,"x":2}\n```', '```json\n{"x":NaN}\n```',
                   '```json\n{}\n```\n```json\n{}\n```']
        for raw in invalid:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                value,_ = service.normalize_json_transport(raw)
                strict_json(value)


if __name__ == "__main__":
    unittest.main()

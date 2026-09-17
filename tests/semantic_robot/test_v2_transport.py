import importlib.util
import ast
from pathlib import Path
import unittest

from semantic_robot.v2.protocol import strict_json

path = Path(__file__).resolve().parents[2]/"scripts/semantic_robot/serve_v2.py"
spec = importlib.util.spec_from_file_location("semantic_v2_service",path)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class TransportTests(unittest.TestCase):
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

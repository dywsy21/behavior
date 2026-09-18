import json
import unittest

from semantic_robot.v2.structured_planning import response_schema, schema_digest
from semantic_robot.v2.harness import Goal, parse_plan
from dataclasses import asdict
from test_v2_transport import service


class StructuredPlanningTests(unittest.TestCase):
    def test_exact_schema_matches_goal_contract_and_isolation(self):
        schema=response_schema("task_plan_v1","plan")
        self.assertEqual(set(schema["items"]["required"]),set(Goal.__dataclass_fields__))
        self.assertFalse(schema["items"]["additionalProperties"])
        self.assertEqual((schema["minItems"],schema["maxItems"]),(1,16))
        schema["items"]["required"].clear()
        self.assertEqual(len(response_schema("task_plan_v1","plan")["items"]["required"]),5)
        self.assertEqual(len(schema_digest()),64)

    def test_whitelist_no_arbitrary_schema_or_finite_action_override(self):
        data={"kind":"plan","system":"s","text":"t","images":[],"allowed":[],"response_schema":"task_plan_v1"}
        self.assertEqual(service.validate_response_schema(data,True),response_schema("task_plan_v1","plan"))
        for changed in ({"response_schema":"unregistered"},{"response_schema":{}},
                        {"kind":"act"},{"allowed":["{}"]},{"extra":True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                service.validate_response_schema({**data,**changed},True)
        with self.assertRaises(ValueError):service.validate_response_schema(data,False)
        old={k:v for k,v in data.items() if k!="response_schema"}
        self.assertIsNone(service.validate_response_schema(old,False))

    def test_schema_is_not_semantic_completeness_or_occupancy_certificate(self):
        # Preserve this limitation: manual pilot checks must verify task coverage.
        nav=[asdict(Goal("navigate","visible table","both","radio visible"))]
        self.assertEqual(len(parse_plan(json.dumps(nav))),1)
        bad=[asdict(Goal("pick","one object","right","held")),
             asdict(Goal("pick","another object","right","held"))]
        with self.assertRaises(ValueError):parse_plan(json.dumps(bad))
        with self.assertRaises(ValueError):parse_plan('Explanation\n```json\n'+json.dumps(nav)+'\n```')

    def test_length_and_boolean_postvalidation_remain_hard(self):
        goal=asdict(Goal("navigate","destination","both","reached"))
        self.assertEqual(len(parse_plan(json.dumps([goal]*16))),16)
        with self.assertRaises(ValueError):parse_plan(json.dumps([goal]*17))
        with self.assertRaises(ValueError):parse_plan(json.dumps([{**goal,"level":1}]))
        with self.assertRaises(ValueError):parse_plan(json.dumps([{**goal,"target":"x"*301}]))


if __name__=="__main__":unittest.main()

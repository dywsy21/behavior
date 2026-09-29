"""JEV-03 CPU contracts: real selection routes, fake network, no simulator."""
import ast
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from semantic_robot.v2.jev_client import JevError
from semantic_robot.v2.jev_control import candidates, authority, validate_actor_ownership
from semantic_robot.v2.jev_policy import JevDecisionPolicy, JevGroundedPolicy, decision_state, load_task_plan
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import HOLD, Action
from test_grounded import setup_controller, grounded_evidence
from test_jev import client, context, Opener

PLAN = Path(__file__).resolve().parents[2]/"configs/semantic_robot/jev_task0_plan.json"


class OwnershipTests(unittest.TestCase):
    def proof(self):
        h, state, bundle, allowed = context()
        ledger = []
        chosen, receipt = JevDecisionPolicy(client(Opener([
            {"intent":"approach"}, {"command":"command_001"}]), journal=ledger.append)).select(
                json.loads(actor_context(h,state,bundle,allowed)),allowed)
        proof = authority(chosen,receipt,allowed)
        return chosen, receipt, allowed, proof, ledger

    def test_whole_current_palette_and_selected_action_are_bound(self):
        chosen, receipt, allowed, proof, _ = self.proof()
        self.assertEqual(proof["command_call"],2)
        for bad_action, bad_allowed in ((HOLD,allowed),(chosen,allowed[::-1]),(chosen,allowed[1:])):
            with self.assertRaises(JevError): authority(bad_action,receipt,bad_allowed)
        for key,value in (("model","not-jev"),("command_call",True),("schema","script"),
                          ("selected_option","command_000"),("action",asdict(HOLD))):
            bad=copy.deepcopy(receipt);bad["result"]["jev_authority"][key]=value
            with self.assertRaises(JevError): authority(chosen,bad,allowed)

    def test_executed_ticks_match_durable_model_choice_not_just_call_counter(self):
        chosen, _, _, proof, ledger = self.proof()
        result = dict(jev_requests=2,controls=3,decisions=[dict(decision=0,control_start=0,control_end=2,
            action=asdict(chosen),accepted_before_motion=True,selection_source="Jev_all_grounded_control",
            jev_authority=proof)])
        ticks=[dict(control=i,decision=0,action23=[0]*23) for i in (1,2)]+[dict(control=3,safety_stop=True)]
        summary=validate_actor_ownership(result,ledger,ticks)
        self.assertEqual(summary["jev_owned_actions"],1)
        self.assertTrue(summary["all_control_ticks_checked"])
        with self.assertRaises(JevError): validate_actor_ownership({**result,"decisions":[]},ledger,ticks)
        bad=copy.deepcopy(ledger);bad[-1]["selections"]["command"]="command_000"
        with self.assertRaises(JevError): validate_actor_ownership(result,bad,ticks)
        for wrong in (ticks[1:],ticks+ticks[-1:],ticks[:1]+[dict(control=2,safety_stop=True)]+ticks[-1:],
                      [dict(control=1,decision=99,action23=[0]*23),*ticks[1:]]):
            with self.assertRaises(JevError):validate_actor_ownership(result,ledger,wrong)
        bad=copy.deepcopy(result);bad["decisions"][0]["selection_source"]="measured_search_controller"
        with self.assertRaises(JevError):validate_actor_ownership(bad,ledger,ticks)
        bad=copy.deepcopy(result);bad["decisions"].append(copy.deepcopy(bad["decisions"][0]))
        with self.assertRaises(JevError):validate_actor_ownership(bad,ledger)

    def test_unrequested_probe_or_wrong_ledger_rejected(self):
        chosen, _, _, proof, ledger = self.proof()
        auth=dict(schema="jev-search-recovery-choice-v1",model="jev-1.13.0",call=3,
                  choice="measure_stationary_reference")
        row=dict(decision=0,control_start=0,control_end=1,action=asdict(chosen),
            accepted_before_motion=True,selection_source="Jev_all_world_search",jev_authority=proof,
            search_reanchor=dict(control_start=1,control_end=13,jev_recovery_authorization=auth))
        result=dict(jev_requests=3,controls=14,decisions=[row])
        ledger.append(dict(call=3,event="validated",selections={"tracking_recovery":auth["choice"]}))
        self.assertEqual(validate_actor_ownership(result,ledger)["jev_requested_probe_controls"],12)
        for bad in (None,{},dict(auth,choice="stop"),dict(auth,call=2)):
            row["search_reanchor"]["jev_recovery_authorization"]=bad
            with self.assertRaises(JevError):validate_actor_ownership(result,ledger)


class RouteTests(unittest.TestCase):
    def test_search_provides_opposing_preflighted_directions_and_no_script_choice(self):
        _, state, h, _, c, depths, receipt = setup_controller()
        h.jev_decision_owner=True
        c.observe(grounded_evidence(visible=False,view="none",target_uv=None),state,depths,receipt)
        with patch.object(c,"search_action",side_effect=AssertionError("script search forbidden")), \
             patch.object(c.search,"propose",side_effect=AssertionError("script ranking forbidden")):
            allowed,mode=candidates(c,state)
        self.assertEqual(mode,"world_search")
        self.assertIn(Action("base","yaw_plus","coarse"),allowed)
        self.assertIn(Action("base","yaw_minus","coarse"),allowed)
        self.assertIn(HOLD,allowed)
        self.assertGreater(len(allowed),2)
        scope=decision_state(json.loads(actor_context(h,state,SimpleNamespace(geometry={}),allowed)),allowed)
        self.assertIn("search",scope["harness"])
        self.assertEqual(scope["harness"]["jev_plan"][0]["target"],"radio")
        turns=[v for v in scope["commands"].values() if v["command"]["move"].startswith("yaw_")]
        self.assertTrue(all(v["current_preflight"]["search_prediction"]["is_prediction_not_measured_coverage"] for v in turns))
        self.assertEqual(c.search.heading,0.)

    def test_search_depth_veto_and_budget_unchanged(self):
        _, state, h, _, c, depths, receipt = setup_controller()
        c.observe(grounded_evidence(visible=False,view="none",target_uv=None),state,depths,receipt)
        c.depth_guard=SimpleNamespace(check=lambda a,*_:(a==HOLD,"TEST_BLOCKED"),receipt=lambda:{})
        self.assertEqual(candidates(c,state)[0],(HOLD,))
        c.search.unknown_budget_travel_m=.36;c.search.search_travel_m=1.
        self.assertEqual(candidates(c,state)[0],())
        self.assertEqual(h.stop_reason,"SEARCH_TRAVEL_BUDGET")

    def test_goal_press_inspection_and_verification_routes_never_pick_winner(self):
        _, state, h, _, c, _, _ = setup_controller()
        c.goal_changed=True
        self.assertEqual(candidates(c,state),((HOLD,),"goal_observation_barrier"))
        self.assertTrue(h.candidate_receipt["tested"][0]["accepted"])
        c.goal_changed=False
        c.press_cycle=SimpleNamespace(owns_selection=True,candidates=Mock(return_value=(HOLD,)))
        self.assertEqual(candidates(c,state),((HOLD,),"press_protocol"))
        c.press_cycle.candidates.assert_called_once_with(state,None,model_selects=True)
        c.press_cycle=None
        with patch.object(type(c),"is_held_search",new_callable=lambda:property(lambda _:True)), \
             patch.object(c,"inspection_candidates",return_value=(HOLD,Action("right","up","micro"))):
            self.assertEqual(candidates(c,state)[1],"held_inspection")
        h.stage="VERIFY_GRASP";h.observation=grounded_evidence(enclosed=True,co_moving=None)
        c.target={"valid":True};c.grasp_verifier=object()
        lift=Action("right","up","fine")
        with patch.object(c,"candidates",return_value=(HOLD,lift,Action("right","up","micro"))), \
             patch.object(c,"verification_action",side_effect=AssertionError("no forced lift")):
            self.assertEqual(candidates(c,state),((HOLD,lift),"grasp_verification"))

    def test_jev_recovery_recommendation_never_queues_macro(self):
        _, _, h, _, c, _, _ = setup_controller()
        h.jev_decision_owner=True;h.observation=grounded_evidence(visible=True)
        for strategy in ("move_forward","scan_left"):
            h.replans=0;c.replan_needed="TEST"
            c.apply_recovery(dict(strategy=strategy,visible_reason="unit"))
            self.assertIsNone(c.reposition);self.assertEqual(c.reposition_left,0)

    def test_plan_and_world_reference_are_model_choices_not_shortcuts(self):
        opener=Opener([{"next_goal":"goal_0"},{"next_goal":"goal_1"},{"reference":"world"}])
        p=JevGroundedPolicy(SimpleNamespace(calls=0,identity={}),client(opener),task_id=0,task_plan=PLAN)
        goals,receipt=p.plan(0,"turn on radio",None)
        self.assertEqual(goals,p.goals)
        self.assertEqual(receipt["result"]["source"],"jev_selected_goal_order")
        with self.assertRaises(JevError):p.plan(0,"again",None)
        h,_,_,_=context();h.goals=goals;h.held_inspection_enabled=h.reference_from_planner=True
        p.resolve_target_reference(h)
        self.assertEqual(p.client.calls,3)
        p.resolve_target_reference(h)
        self.assertEqual(p.client.calls,3)  # exact goal reference already bound

    def test_dependency_is_hard_constraint_not_prompt_only(self):
        opener=Opener([{"next_goal":"goal_1"}])
        p=JevGroundedPolicy(SimpleNamespace(calls=0,identity={}),client(opener),task_id=0,task_plan=PLAN)
        with self.assertRaises(JevError):p.plan(0,"turn on radio",None)
        options=json.loads(opener.requests[0][0].data)["questions"]["next_goal"]["criteria"]
        self.assertNotIn("goal_1",options)
        h,_,_,_=context();h.goals=p.goals;h.index=1
        h.held_inspection_enabled=h.reference_from_planner=True
        self.assertIsNone(p.resolve_target_reference(h))
        self.assertEqual(h.stop_reason,"TASK_DEPENDENCY_REQUIRES_VERIFIED_HOLD")
        self.assertNotIn(1,h.target_references)

    def test_invalid_dependency_graph_or_held_reference_never_reaches_model(self):
        original=json.loads(PLAN.read_text())
        for fields in (dict(dependencies=[[1],[0]]),dict(dependencies=[[],[]]),
                       dict(dependencies=[[],[True]]),dict(target_references=["world","held_left"]),
                       dict(dependencies=[[],[0,0]]),dict(target_references=["world","guessed"])):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"plan.json"
                path.write_text(json.dumps({**original,**fields}))
                with self.assertRaises(ValueError):load_task_plan(path,0)

    def test_held_part_can_never_bind_world_even_with_another_verified_load(self):
        opener=Opener([{"reference":"world"}])
        p=JevGroundedPolicy(SimpleNamespace(calls=0,identity={}),client(opener),task_id=0,task_plan=PLAN)
        h,_,_,_=context();h.goals=p.goals;h.index=1
        h.held_inspection_enabled=h.reference_from_planner=True
        h.hold_verified["right"]=True;h.held["right"]="radio"
        with self.assertRaises(JevError):p.resolve_target_reference(h)
        self.assertNotIn("world",json.loads(opener.requests[0][0].data)["questions"]["reference"]["criteria"])
        self.assertNotIn(1,h.target_references)

    def test_offline_search_loader_rejects_unpinned_files_before_api(self):
        from probe_jev_all import search_input
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"robot_calibration.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"Frozen search source changed"):search_input(folder)

    def test_offline_probe_credential_rules_checked_before_source_or_client(self):
        import probe_jev_all as probe
        args=["probe","--search-source","not-read","--output","not-created"]
        for extra,env in (([],{}),([],dict(TYPESAFE_API_KEY="dummy-forbidden")),
                          (["--dry-run","--key-file","not-read"],{})):
            with patch.dict(os.environ,env,clear=True),patch("sys.argv",args+extra), \
                 patch.object(probe,"search_input") as source,patch.object(probe,"JevClient") as api:
                with self.assertRaises(ValueError):probe.main()
                source.assert_not_called();api.assert_not_called()

    def test_actual_runner_jev_branch_cannot_fall_through_to_old_script(self):
        path=Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py"
        tree=ast.parse(path.read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="args.controller == 'jev'"
                    and any(isinstance(x,ast.ImportFrom) and x.module=="semantic_robot.v2.jev_control" for x in n.body))
        source=ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
        self.assertIn("select_feasible_action",source)
        self.assertIn("jev_authority(action, call, allowed)",source)
        self.assertIn("bind_model_choice(action, controls)",source)
        for forbidden in ("search_action(","verification_action(","allowed[0]","action = HOLD"):
            self.assertNotIn(forbidden,source)


if __name__=="__main__":unittest.main()

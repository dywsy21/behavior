from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from behavior_light_state import capture_lights,restore_lights,validate_light_restore
from behavior_branch_state import capture_branch_metadata,restore_branch_metadata
from test_behavior_branch_state import fixture


class LightToggleSynchronizer:
    def __init__(self, scene):
        self.scene=scene
        self._toggle_values=dict(switch=True,lamp=False)
        self._target_visibility=dict(room=False,lamp=True)
    def _get_toggle_values(self):return self.scene.toggles


def lights():
    objs=[SimpleNamespace(name=n,category=c,in_rooms=['living'],visible=v,prims={'/lamp/light':'inherited'})
          for n,c,v in [('room','room_light',False),('lamp','table_lamp',True),('switch','electric_switch',True)]]
    scene=SimpleNamespace(objects=objs,toggles=dict(switch=True,lamp=False))
    return SimpleNamespace(light_synchronizer=LightToggleSynchronizer(scene),env_accessor=SimpleNamespace(scene=scene)),objs


def apply(obj,value):
    if obj.category=='room_light':obj.visible=value
    else:obj.prims={k:'inherited' if value else 'invisible' for k in obj.prims}


class LightTests(unittest.TestCase):
    @patch('behavior_light_state._self_visibility',side_effect=lambda obj:dict(obj.prims))
    @patch('behavior_light_state._apply_visibility',side_effect=apply)
    def test_restores_actual_light_and_edge_history_without_all_on_reset(self,*_):
        inst,objs=lights();saved=capture_lights(inst)
        objs[0].visible=True;objs[1].prims={'/lamp/light':'invisible'}
        inst.light_synchronizer._toggle_values={'switch':False,'lamp':True}
        inst.light_synchronizer._target_visibility={'room':True,'lamp':False}
        restore_lights(inst,saved)
        self.assertEqual(capture_lights(inst),saved)
        self.assertFalse(objs[0].visible)
        self.assertEqual(inst.light_synchronizer._toggle_values,dict(switch=True,lamp=False))

    @patch('behavior_light_state._self_visibility',side_effect=lambda obj:dict(obj.prims))
    def test_stale_world_or_cross_scene_or_missing_target_reject(self,*_):
        inst,objs=lights();saved=capture_lights(inst)
        for field in ('toggle_values','target_visibility','self_prim_visibility','topology'):
            bad=deepcopy(saved);bad[field]={}
            with self.assertRaises(ValueError):validate_light_restore(inst,bad)
            self.assertFalse(objs[0].visible)
        inst.env_accessor.scene.toggles['switch']=False
        with self.assertRaises(ValueError):validate_light_restore(inst,saved)
        with self.assertRaises(ValueError):capture_lights(inst)

    @patch('behavior_light_state._self_visibility',side_effect=lambda obj:dict(obj.prims))
    @patch('behavior_light_state._apply_visibility',side_effect=apply)
    def test_global_metadata_validation_is_atomic_and_legacy_does_not_drop_lights(self,*_):
        ev=fixture();inst,objs=lights();old_scene=ev.instance_eval_states[0].env_accessor.scene
        # Keep the original hashable scene key used by metrics.
        class Scene:pass
        scene=Scene();scene.objects=objs;scene.toggles=dict(switch=True,lamp=False)
        live=ev.instance_eval_states[0];live.env_accessor.scene=scene
        live.light_synchronizer=LightToggleSynchronizer(scene)
        for metric in live.metrics:metric.state={scene:metric.state[old_scene]}
        saved=capture_branch_metadata(ev);self.assertEqual(saved['schema'],'behavior_prefix_branch_metadata_v2')
        bad=deepcopy(saved);del bad['light_synchronizer'];ev.env.env._current_steps=[999]
        with self.assertRaises(ValueError):restore_branch_metadata(ev,bad)
        self.assertEqual(ev.env.env._current_steps,[999])
        restore_branch_metadata(ev,saved);self.assertEqual(capture_branch_metadata(ev),saved)


if __name__=='__main__':unittest.main()

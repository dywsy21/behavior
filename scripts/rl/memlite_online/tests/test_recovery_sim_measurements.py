"""Structural sensor tests; not a claim of real simulator acceptance."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest
from skill_aligned_reward import SkillIdentity
from skill_sim_measurements import OmniSkillMeasurements


class Inside: pass
class OnTop: pass
class Open: pass


class Object:
    def __init__(self,name):
        self.name=name;self.aabb=([0.,0.,0.],[.1,.1,.1]);self.velocity=np.zeros(3)
        self.states={Inside:SimpleNamespace(get_value=lambda _:True),OnTop:SimpleNamespace(get_value=lambda _:True)}
    def get_position_orientation(self):return np.zeros(3),[0,0,0,1]
    def get_linear_velocity(self):return self.velocity


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.obj=Object('basket_27');self.dest=Object('table_8')
        self.contact={'left':'FALSE','right':'FALSE'}
        self.robot=SimpleNamespace(arm_names=['left','right'],scene=SimpleNamespace(objects=[self.obj,self.dest]),
            is_grasping=lambda arm,candidate_obj:SimpleNamespace(name=self.contact[arm]),
            get_eef_position=lambda arm:np.zeros(3))
        self.accessor=SimpleNamespace(robot=self.robot,object_scope={'basket.n.01_1':self.obj,'table.n.02_1':self.dest})

    def sensor(self,verb,**extra):
        skill=dict(verb=verb,target=self.obj.name,arm='LEFT',**extra)
        ident=SkillIdentity('session','task',1,'ep','ctx',digest([skill]),0)
        return OmniSkillMeasurements(self.accessor,ident,skill),ident

    def test_grasp_uses_exact_requested_hand_and_not_nearest_class(self):
        sensor,ident=self.sensor('GRASP');self.contact['right']='TRUE'
        self.assertFalse(sensor.read(ident)[0]['achieved'])
        self.contact['left']='TRUE'
        self.assertTrue(sensor.read(ident)[0]['achieved'])
        self.assertFalse(sensor.read(ident)[1]['actor_input'])
        with self.assertRaises(ValueError):sensor.resolve('basket')
        self.robot.scene.objects.append(Object('basket_27'))
        with self.assertRaises(ValueError):sensor.read(ident)

    def test_placement_requires_official_relation_release_and_stability(self):
        for verb in ('PLACE_IN','PLACE_ON'):
            sensor,ident=self.sensor(verb,destination='table_8')
            self.contact['left']='TRUE'
            self.assertFalse(sensor.read(ident)[0]['achieved'])
            self.contact['left']='FALSE';self.obj.velocity=np.array([.2,0.,0.])
            self.assertFalse(sensor.read(ident)[0]['achieved'])
            self.obj.velocity=np.zeros(3)
            self.assertTrue(sensor.read(ident)[0]['achieved'])

    def test_unknown_contact_and_cross_session_never_become_negatives(self):
        sensor,ident=self.sensor('GRASP')
        with self.assertRaises(ValueError):sensor.read(replace(ident,episode='another'))
        self.contact['left']='UNKNOWN'
        with self.assertRaises(ValueError):sensor.read(ident)

    def test_functional_opening_not_five_percent_or_time(self):
        position=[.05]
        contract=(SimpleNamespace(get_value=lambda:True),SimpleNamespace(get_state=lambda:position),1.,0.)
        with patch('skill_sim_measurements.articulation_contract',return_value=contract):
            sensor,ident=self.sensor('OPEN_DOOR')
        self.assertFalse(sensor.read(ident)[0]['achieved'])
        position[0]=.4
        self.assertTrue(sensor.read(ident)[0]['achieved'])

    def test_no_unverified_navigation_or_changed_scene_binding(self):
        with self.assertRaises(ValueError):self.sensor('NAVIGATE')
        sensor,ident=self.sensor('GRASP')
        self.accessor.robot=SimpleNamespace()
        with self.assertRaises(ValueError):sensor.read(ident)

    def test_wrong_skill_member_or_bundle_binding_fails_before_measurement(self):
        sensor,ident=self.sensor('GRASP')
        with self.assertRaisesRegex(ValueError,'semantic bundle'):
            OmniSkillMeasurements(self.accessor,replace(ident,bundle_sha256='b'*64),sensor.skill)
        with self.assertRaises(ValueError):
            OmniSkillMeasurements(self.accessor,replace(ident,member=1),sensor.skill)


if __name__=='__main__':unittest.main()

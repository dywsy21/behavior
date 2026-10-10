from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_expert_feedback import expert_unknown_feedback
from g05.utils.memlite_skill_protocol import canonical_json,semantic_active_skills_text
from g05.utils.memlite_causal_feedback import FeedbackIdentity,CausalFeedbackLedger


def episode():
    def segment(target,parent):
        skill=dict(verb='GRASP',target=target,source='',destination='',target_part='',arm='LEFT',unbound_relation='')
        return dict(semantic=canonical_json([skill]),text=semantic_active_skills_text([skill]),parent=parent)
    segments=[segment('cup','p'),segment('plate','q'),segment('future','future goal')]
    a,b=segments[:2]
    anchors=[(3,0,'None','None',[]),(19,0,a['text'],'p',[]),(35,1,a['text'],'p',[]),
             (51,1,b['text'],'q',[]),(67,0,b['text'],'q',[]),(83,2,a['text'],'p',[])]
    return dict(segments=segments,anchors=anchors)


class ExpertFeedbackTests(unittest.TestCase):
    def test_replay_matches_runtime_ledger_and_keeps_unknown(self):
        ep=episode();ident=FeedbackIdentity('test','task',1,'episode','a'*64,'b'*64)
        ledger=CausalFeedbackLedger(ident)
        for index,(step,segment_index,*_) in enumerate(ep['anchors']):
            actual=expert_unknown_feedback(ep,index)
            self.assertEqual(actual,ledger.projection(ident,step))
            if actual!='none':
                x=json.loads(actual)
                self.assertEqual(x['estimated_bundle_outcome'],'UNKNOWN')
                self.assertTrue(all(m['confidence']==0 for m in x['estimated_member_outcomes']))
            segment=ep['segments'][segment_index]
            ledger.issued(ident,step,segment['semantic'],segment['parent'])
        self.assertEqual(json.loads(expert_unknown_feedback(ep,2))['same_intent_controls'],32)
        self.assertEqual(json.loads(expert_unknown_feedback(ep,2))['same_intent_planner_refreshes'],1)
        self.assertEqual(json.loads(expert_unknown_feedback(ep,5))['attempt_index'],2)

    def test_current_target_and_future_cannot_change_prior_feedback(self):
        ep=episode();before=expert_unknown_feedback(ep,5)
        ep['segments'][2]['parent']='changed current target'
        ep['segments'][2]['semantic']='DO NOT READ CURRENT TARGET'
        ep['anchors'].append((99,2,'future-only text','future',[]))
        self.assertEqual(expert_unknown_feedback(ep,5),before)

    def test_rejects_broken_history_and_bad_clock(self):
        for key in ('initial','prior','clock'):
            ep=episode();anchors=ep['anchors']
            if key=='initial':anchors[0]=(3,0,'invented','None',[])
            if key=='prior':anchors[2]=(35,1,'wrong previous','p',[])
            if key=='clock':anchors[1]=(3,0,anchors[1][2],'p',[])
            with self.subTest(key=key),self.assertRaises(ValueError):expert_unknown_feedback(ep,5)


if __name__=='__main__':unittest.main()

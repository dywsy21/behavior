"""Observable command counters on original successful-demonstration inputs.

The command trace is the SAME annotation-derived, stride16 causal trace used
by Stage1Dataset for previous intent/memory. It is not a new physical outcome
label or an on-policy rollout. Only already-issued anchors precede a query.
"""
from collections import OrderedDict
import json

from recovery_observer_training import feedback_text


def expert_unknown_feedback(episode, anchor_index):
    anchors=episode['anchors'];segments=episode['segments']
    if type(anchor_index) is not int or not 0<=anchor_index<len(anchors):
        raise ValueError('Invalid original command anchor')
    frame,_,previous,parent,_=anchors[anchor_index]
    attempts=OrderedDict();last_key=None;last_text=None;started=refreshes=attempt=0;last_step=-1
    for step,segment_index,prior_intent,prior_parent,_ in anchors[:anchor_index]:
        if step<=last_step or step>=frame:
            raise ValueError('Original command history is not strictly prior')
        if ((last_key is None and (prior_intent!='None' or prior_parent!='None'))
                or (last_key is not None and (prior_intent!=last_text or prior_parent!=last_key[1]))):
            raise ValueError('Original previous-command history drift')
        segment=segments[segment_index];key=(segment['semantic'],segment['parent'])
        if key==last_key:
            refreshes+=1
        else:
            attempts[key]=attempts.get(key,0)+1;attempts.move_to_end(key)
            if len(attempts)>64:attempts.popitem(last=False)
            started,refreshes,attempt=step,0,attempts[key]
        last_key,last_text,last_step=key,segment['text'],step
    if last_key is None:
        if previous!='None' or parent!='None':raise ValueError('Initial sample has invented prior command')
        return 'none'
    if previous!=last_text or parent!=last_key[1]:
        raise ValueError('Feedback and model previous intent differ')
    members=json.loads(last_key[0])
    if not members:raise ValueError('No actual previous command members')
    return feedback_text(dict(served_controls=frame-started,repeated_planning_count=refreshes,attempt_number=attempt),
        [dict(member=i,estimated_outcome='UNKNOWN',confidence=0.) for i in range(len(members))])


def with_expert_feedback(dataset_class):
    """Separate opt-in wrapper; the main stage1 reader is not changed."""
    class CausalFeedbackExpert(dataset_class):
        def raw(self,index,*,images=True):
            if self.branch!='high':raise ValueError('Planner feedback must not change the low SFT input')
            raw,identity=super().raw(index,images=images)
            serial,anchor=(int(x) for x in self.candidates[index])
            raw['model_projection']['execution_feedback']=expert_unknown_feedback(self.episode(serial),anchor)
            return raw,identity
    return CausalFeedbackExpert

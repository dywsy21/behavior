"""Veto repeated ineffective base approaches using observed execution progress.

No object-motion/collision truth: check the static-target assumption used by the
candidate ranking against fresh robot-relative RGB-D contact observations.
"""
import numpy as np
from .protocol import TRANSLATIONS


class ApproachProgress:
    def __init__(self):
        self.previous=None;self.window=[];self.blocked={};self.goal=None
        self.last_execution=None;self.events=[]

    def allowed(self,action):
        return not (action.part=="base" and action.move in self.blocked)

    def observe(self,*,goal_index,kind,stage,points,poses,execution,action,feedback,motion,loaded=False):
        if goal_index!=self.goal:
            self.goal=goal_index;self.previous=None;self.window=[];self.blocked={};self.last_execution=None
        self.events=[]
        if execution==self.last_execution:return self.context()
        self.last_execution=execution
        old=self.previous
        distance=(float(np.mean([np.linalg.norm(points[a]-poses[a][:3,3]) for a in points])) if points else None)
        self.previous={"points":{a:np.asarray(p).copy() for a,p in points.items()},"distance":distance,
                       "kind":kind,"stage":stage,"execution":execution}
        body=np.asarray(motion.get("body_transform_current_in_previous",np.eye(4)),dtype=float)
        measured=bool(motion.get("valid") and "body_transform_current_in_previous" in motion
                      and body.shape==(4,4) and np.isfinite(body).all())
        # A new pixel or OPEN cannot forget the latch. Require an actual changed
        # hand pose, measured retreat, or new approach angle before retrying.
        for move,record in list(self.blocked.items()):
            changed=False
            for arm,anchor in record["poses"].items():
                if arm not in poses:continue
                now=poses[arm]
                angle=np.arccos(np.clip((np.trace(anchor[:3,:3].T@now[:3,:3])-1)/2,-1,1))
                changed|=np.linalg.norm(now[:3,3]-anchor[:3,3])>=.01 or angle>=np.deg2rad(5)
            if measured and old is not None:
                record["body_transform"]=record["body_transform"]@body
                travelled=record["body_transform"];offset=travelled[:3,3]
                yaw=float(np.arctan2(travelled[1,0],travelled[0,0]))
                direction=np.asarray(TRANSLATIONS[move]);advance=offset@direction
                side=offset-advance*direction
                changed|=advance<=-.04 or np.linalg.norm(side[:2])>=.04 or abs(yaw)>=np.deg2rad(10)
            if changed:
                del self.blocked[move];self.window=[]
                self.events.append({"event":"NEW_MEASURED_APPROACH_POSE","released_direction":move})
        eligible=(old is not None and kind=="pick" and old["kind"]=="pick" and
            stage in ("ALIGN","APPROACH") and old["stage"] in ("ALIGN","APPROACH") and
            not loaded and action is not None and action.part=="base" and
            action.move in ("forward","back","left","right") and
            feedback is not None and feedback.get("status")=="TARGET_REACHED" and measured and
            # Failure of predicted progress matters at any observed distance.
            # The 0.10 m cutoff missed repeated advances at 0.11-0.15 m, even
            # though the same measured static-target prediction kept failing.
            distance is not None and old["distance"] is not None and
            set(points)==set(old["points"]) and body[:3,3]@np.asarray(TRANSLATIONS[action.move])>=.012)
        if not eligible:self.window=[];return self.context()
        inv=np.linalg.inv(body)
        expected=float(np.mean([np.linalg.norm(inv[:3,:3]@old["points"][a]+inv[:3,3]-poses[a][:3,3]) for a in points]))
        gain=old["distance"]-expected
        if gain<.005:self.window=[];return self.context()
        if self.window and self.window[-1]["move"]!=action.move:self.window=[]
        self.window.append({"execution":execution,"move":action.move,"before_m":old["distance"],
            "after_m":distance,"expected_gain_m":gain,"measured_base_travel_m":float(np.linalg.norm(body[:2,3]))})
        self.window=self.window[-3:]
        expected_gain=sum(r["expected_gain_m"] for r in self.window)
        reduction=self.window[0]["before_m"]-distance
        if len(self.window)>=3 and expected_gain>=.025 and reduction<max(.003,.2*expected_gain):
            record={"event":"REPEATED_BASE_APPROACH_WITHOUT_CONTACT_PROGRESS","direction":action.move,
                "executions":[r["execution"] for r in self.window],"expected_static_target_gain_m":expected_gain,
                "observed_contact_error_reduction_m":reduction,
                "measured_base_travel_m":sum(r["measured_base_travel_m"] for r in self.window),
                "not_proof_of_collision_or_object_motion":True}
            self.events.append(record)
            self.blocked[action.move]={"evidence":record,"poses":{a:p.copy() for a,p in poses.items()},
                "body_transform":np.eye(4)}
            self.window=[]
        return self.context()

    def context(self):
        return {"source":"observed_onboard_approach_progress","scene_truth":False,
            "blocked_base_directions":sorted(self.blocked),"events":list(self.events),
            "evidence":[r["evidence"] for r in self.blocked.values()],
            "recent_measured_attempts":list(self.window),
            "instruction":"Do not repeat a blocked base direction. Use a different feasible hand pose or retreat; predicted point-distance gain assumed a stationary target and did not materialize."}

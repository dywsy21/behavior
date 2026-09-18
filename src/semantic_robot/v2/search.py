"""Bounded egocentric search coverage from executed odometry and actual views.

No world pose, scene map, target location, or command-issued-as-motion shortcuts.
"""
import math
import numpy as np

from .protocol import Action


def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


class CoverageSearch:
    bins = 24

    def __init__(self):
        self.goal_index=None
        self.heading=0.
        self.xy=np.zeros(2)
        self.origin=np.zeros(2)
        self.covered=set()
        self.nodes=[]
        self.direction=1
        self.new_coverage=False
        self.observed_heading=0.
        self.travel_m=0.
        self.rotation_rad=0.
        self.last_seen_heading=None
        self.misses=0

    def executed(self, feedback):
        # Use actual measured displacement, including failed/partial motions.
        delta=np.asarray(feedback.get("base_integral",[0.,0.,0.]),dtype=float)
        if delta.shape!=(3,) or not np.isfinite(delta).all():
            raise ValueError("Finite measured base integral required")
        c,s=math.cos(self.heading+delta[2]/2),math.sin(self.heading+delta[2]/2)
        self.xy += np.array([[c,-s],[s,c]])@delta[:2]
        self.heading += float(delta[2])
        self.travel_m += float(np.linalg.norm(delta[:2]))
        self.rotation_rad += abs(float(delta[2]))

    def observe(self, goal_index, camera_pose, K, width, valid_depth_fraction, visible):
        if goal_index!=self.goal_index:
            self.goal_index=goal_index
            self.covered=set(); self.nodes=[]; self.origin=self.xy.copy()
            self.last_seen_heading=None; self.misses=0
        if np.linalg.norm(self.xy-self.origin)>.25:
            self.nodes.append({"xy":self.origin.tolist(),"bins":sorted(self.covered)})
            self.nodes=self.nodes[-8:]
            self.origin=self.xy.copy(); self.covered=set()
            for old in self.nodes:
                if np.linalg.norm(self.origin-old["xy"])<.20:
                    self.covered.update(old["bins"])
        look=-np.asarray(camera_pose)[:3,2]
        self.observed_heading=self.heading+math.atan2(look[1],look[0])
        before=len(self.covered)
        if valid_depth_fraction>=.25 and np.linalg.norm(look[:2])>.4:
            half_fov=.7*math.atan(width/(2*float(K[0][0])))
            for index in range(self.bins):
                center=2*math.pi*index/self.bins
                if abs(wrap(center-self.observed_heading))<=half_fov:
                    self.covered.add(index)
        self.new_coverage=len(self.covered)>before
        if visible:
            self.last_seen_heading=self.observed_heading; self.misses=0
        else:
            self.misses+=1
        return self.new_coverage

    @property
    def complete(self):
        return len(self.covered)==self.bins

    def propose(self):
        if self.travel_m>1.2 or self.rotation_rad>4*math.pi:
            return None,"SEARCH_TRAVEL_BUDGET"
        if self.complete:
            return None,"LOCAL_VIEW_SWEEP_COMPLETE_TARGET_NOT_FOUND"
        # Reacquire a just-lost target, then commit to a monotonically expanding
        # sweep. Tiny alternating turns cannot repeatedly count as new coverage.
        if self.last_seen_heading is not None and self.misses<=2:
            error=wrap(self.last_seen_heading-self.observed_heading)
            if abs(error)>math.radians(5):
                direction=1 if error>0 else -1
                return Action("base","yaw_plus" if direction>0 else "yaw_minus","coarse"),"REACQUIRE_LAST_OBSERVED_BEARING"
        return Action("base","yaw_plus" if self.direction>0 else "yaw_minus","coarse"),"EXPAND_OBSERVED_HEADING_COVERAGE"

    def context(self):
        return {"frame":"integrated_body_odometry_not_global_truth",
                "observed_heading_deg":round(math.degrees(self.observed_heading),2),
                "covered_heading_bins":sorted(self.covered),"total_bins":self.bins,
                "new_observation_coverage":self.new_coverage,"sweep_complete":self.complete,
                "previous_viewpoints":len(self.nodes),"travel_m":round(self.travel_m,3),
                "rotation_deg":round(math.degrees(self.rotation_rad),2),
                "coverage_is_not_goal_completion":True}

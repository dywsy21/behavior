"""Private RGB-D sensor adapter matching the official allowed modalities.

Reads only three robot-mounted cameras. Does not touch object/scene queries.
The legacy evaluator's RGB-only preprocessing is deliberately left unchanged.
"""
import hashlib
import numpy as np

from semantic_robot.og_backend import array
from .grounding import validate_depth


class OnboardRGBD:
    def __init__(self, env):
        robot=env.robots[0]
        self.sensors={}
        parents={"head":"zed_link", "left_wrist":"left_realsense_link", "right_wrist":"right_realsense_link"}
        for view,parent in parents.items():
            matches=[sensor for name,sensor in robot.sensors.items()
                     if parent in name and hasattr(sensor,"intrinsic_matrix")]
            if len(matches)!=1:
                raise ValueError("Unique robot-mounted RGB-D camera required: "+view)
            sensor=matches[0]
            # OfficialEvaluatorSession ALREADY installs RGBDFullResWrapper.
            # Reassigning even the SAME resolution destroys/recreates render
            # products and can invalidate initialized PhysX articulation views.
            # This adapter is strictly read-only, not another sensor wrapper.
            expected=720 if view=="head" else 480
            if set(sensor.modalities)!={"rgb","depth_linear"}:
                raise ValueError("Official RGB-D wrapper must configure modalities before robot initialization")
            if (sensor.image_height,sensor.image_width)!=(expected,expected):
                raise ValueError("Official RGB-D resolution mismatch; no live sensor reconfiguration")
            self.sensors[view]=sensor
        self.snapshot_id=0

    def read(self, model, *, render):
        # Reset / teleports can leave annotator buffers at the PRE-reset pose.
        # Do not trust get_obs() alone. Render-only updates flush the same
        # asynchronous sensor pipeline used by the official light synchronizer.
        # No env.step / physics action / object query is permitted in this hook.
        if not callable(render):
            raise ValueError("Explicit current-frame render barrier required")
        for _ in range(4):render()
        self.snapshot_id+=1
        images,depths,receipt={},{},{}
        for view,sensor in self.sensors.items():
            sample,_=sensor.get_obs()
            # Enforce the whitelist on the interface, even if an engine version
            # ever supplies extra buffers. No info/segmentation is forwarded.
            rgb=array(sample["rgb"])
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1] not in (3,4):
                raise ValueError("Invalid onboard uint8 RGB")
            rgb=rgb[...,:3].copy()
            camera=model.spec["metadata"]["cameras"][view]
            depth=validate_depth(array(sample["depth_linear"]),camera).copy()
            if rgb.shape[:2]!=depth.shape:
                raise ValueError("RGB/depth alignment or resolution drift")
            valid=np.isfinite(depth)&(depth>.025)&(depth<4.)
            images[view+"_rgb"]=rgb.transpose(2,0,1)
            depths[view]=depth
            receipt[view]={"modalities":["rgb","depth_linear"],"depth_units":"metres",
                           "depth_convention":"distance_to_image_plane",
                           "shape":list(depth.shape),"valid_fraction":float(valid.mean()),
                           "rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),
                           "depth_sha256":hashlib.sha256(depth.tobytes()).hexdigest(),
                           "same_sensor_current_render":True,"render_barrier_updates":4,
                           "snapshot_id":self.snapshot_id,"control_steps_in_capture":0}
        return images,depths,receipt

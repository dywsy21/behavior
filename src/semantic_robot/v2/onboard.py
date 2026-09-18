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
        from omnigibson.eval.utils.eval_utils import set_sensor_modalities
        robot=env.robots[0]
        self.sensors={}
        parents={"head":"zed_link", "left_wrist":"left_realsense_link", "right_wrist":"right_realsense_link"}
        for view,parent in parents.items():
            matches=[sensor for name,sensor in robot.sensors.items()
                     if parent in name and hasattr(sensor,"intrinsic_matrix")]
            if len(matches)!=1:
                raise ValueError("Unique robot-mounted RGB-D camera required: "+view)
            sensor=matches[0]
            set_sensor_modalities(sensor,{"rgb","depth_linear"})
            sensor.image_height=sensor.image_width=720 if view=="head" else 480
            self.sensors[view]=sensor
        env.load_observation_space()

    def read(self, model):
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
                           "same_sensor_current_render":True}
        return images,depths,receipt

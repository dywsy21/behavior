"""Private RGB-D sensor adapter matching the official allowed modalities.

Reads only three robot-mounted cameras. Does not touch object/scene queries.
The legacy evaluator's RGB-only preprocessing is deliberately left unchanged.
"""
import hashlib
import numpy as np

from semantic_robot.og_backend import array
from .grounding import validate_depth


class OnboardRGBD:
    def __init__(self, env, *, resolutions=None):
        # Explicit experiment profiles may change resolution before the robot
        # is initialized. Never infer an allowed shape from a live camera and
        # thereby silently accept a mismatched/legacy calibration.
        expected_sizes = {"head":720, "left_wrist":480, "right_wrist":480}
        if resolutions is not None:
            if (type(resolutions) is not dict or set(resolutions) != set(expected_sizes) or
                    any(type(size) is not int or size <= 0 for size in resolutions.values())):
                raise ValueError("Exact three positive integer camera resolutions required")
            expected_sizes = resolutions.copy()
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
            expected=expected_sizes[view]
            if set(sensor.modalities)!={"rgb","depth_linear"}:
                raise ValueError("Official RGB-D wrapper must configure modalities before robot initialization")
            if (sensor.image_height,sensor.image_width)!=(expected,expected):
                raise ValueError("Registered RGB-D resolution mismatch; no live sensor reconfiguration")
            self.sensors[view]=sensor
        self.snapshot_id=0

    def read(self, model, *, render, synchronize=None):
        # Reset / teleports can leave annotator buffers at the PRE-reset pose.
        # Do not trust get_obs() alone. Render-only updates flush the same
        # asynchronous sensor pipeline used by the official light synchronizer.
        # No env.step / physics action / object query is permitted in this hook.
        if not callable(render):
            raise ValueError("Explicit current-frame render barrier required")
        if synchronize is None:
            for _ in range(4):render()
            time_receipt = None
        else:
            if not callable(synchronize): raise ValueError('Callable native synchronization required')
            time_receipt = synchronize()
            if (set(time_receipt) != set(self.sensors) or
                    any(r.get('reference_time_verified') is not True or r.get('physics_ticks_in_capture') != 0
                        for r in time_receipt.values())):
                raise ValueError('Incomplete synchronized RGB-D receipt')
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
            receipt[view]['render_barrier_updates'] = 4 if time_receipt is None else 1
            receipt[view]['freshness_proof'] = 'legacy_fixed_updates_unverified' if time_receipt is None else 'native_reference_time'
            if time_receipt is not None: receipt[view]['native_time'] = time_receipt[view]
        return images,depths,receipt

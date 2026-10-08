"""Full native camera resolutions, RGB/proprio only; no state manipulation."""
from omnigibson.envs import EnvironmentWrapper


class RGBOnlyFullResWrapper(EnvironmentWrapper):
    @classmethod
    def camera_spec(cls):
        return dict(modalities=['rgb'], resolution={
            'head': (720,720), 'left_wrist': (480,480), 'right_wrist': (480,480)})

    def __init__(self, env):
        super().__init__(env=env)

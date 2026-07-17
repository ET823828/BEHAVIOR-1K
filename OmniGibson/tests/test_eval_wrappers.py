from types import SimpleNamespace

from omnigibson.eval.utils.eval_utils import HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.eval.wrappers import RGBDFullResWrapper


class _FakeVisionSensor:
    def __init__(self, name):
        self.name = name
        self.modalities = {"rgb"}
        self.image_height = 224
        self.image_width = 224
        self.observation_space = object()

    def add_modality(self, modality):
        self.modalities.add(modality)

    def remove_modality(self, modality):
        self.modalities.remove(modality)

    def load_observation_space(self):
        return self.observation_space


def test_rgbd_full_res_wrapper_only_reloads_camera_spaces():
    sensor_names = {
        "head": "zed_link:Camera:0",
        "left_wrist": "left_realsense_link:Camera:0",
        "right_wrist": "right_realsense_link:Camera:0",
    }
    sensors = {name: _FakeVisionSensor(name) for name in sensor_names.values()}
    robot = SimpleNamespace(name="robot_r1", sensors=sensors)
    robot_space = SimpleNamespace(spaces={name: object() for name in sensors})

    def fail_if_full_environment_space_is_reloaded():
        raise AssertionError("wrapper must not reload proprioception while physics handles are unavailable")

    env = SimpleNamespace(
        robots=[robot],
        _eval_robot_config={"camera_sensor_names": {role: f"robot_r1:{name}" for role, name in sensor_names.items()}},
        observation_space=SimpleNamespace(spaces={robot.name: robot_space}),
        load_observation_space=fail_if_full_environment_space_is_reloaded,
    )

    RGBDFullResWrapper(env)

    for role, sensor_name in sensor_names.items():
        sensor = sensors[sensor_name]
        assert sensor.modalities == {"rgb", "depth_linear"}
        expected_resolution = HEAD_RESOLUTION if role == "head" else WRIST_RESOLUTION
        assert (sensor.image_height, sensor.image_width) == expected_resolution
        assert robot_space.spaces[sensor_name] is sensor.observation_space

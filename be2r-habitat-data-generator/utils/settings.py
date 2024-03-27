import json
import os
import magnum as mn

from typing import Any, Dict

import habitat_sim
import habitat_sim.agent
from habitat_sim.bindings import built_with_bullet

from src import actions

# TODO: add noise model and znear

BLACK = mn.Color4.from_linear_rgb_int(0)

# [default_sim_settings]
default_sim_settings: Dict[str, Any] = {
    # path to .scene_dataset.json file
    "scene_dataset_config_file": "default",
    # name of an existing scene in the dataset, a scene, stage, or asset filepath, or "NONE" for an empty scene
    "scene": "NONE",
    # camera sensor parameters
    "width": 640,
    "height": 480,
    # horizontal field of view in degrees
    "hfov": 90,
    # far clipping plane
    "zfar": 1000.0,
    # optional background color override for rgb sensors
    "clear_color": BLACK,
    # vertical offset of the camera from the agent's root position (e.g. height of eyes)
    "sensor_height": 1.5,
    # defaul agent ix
    "default_agent": 0,
    # radius of the agent cylinder approximation for navmesh
    "agent_radius": 0.1,
    # pick sensors to use
    "color_sensor": True,
    "semantic_sensor": False,
    "depth_sensor": False,
    "ortho_rgba_sensor": False,
    "ortho_depth_sensor": False,
    "ortho_semantic_sensor": False,
    "fisheye_rgba_sensor": False,
    "fisheye_depth_sensor": False,
    "fisheye_semantic_sensor": False,
    "equirect_rgba_sensor": False,
    "equirect_depth_sensor": False,
    "equirect_semantic_sensor": False,
    # random seed
    "seed": 1,
    # path to .physics_config.json file
    "physics_config_file": "data/default.physics_config.json",
    # use bullet physics for dyanimcs or not - make default value whether or not
    # Simulator was built with bullet enabled
    "enable_physics": built_with_bullet,
    # ensure or create compatible navmesh for agent paramters
    "default_agent_navmesh": True,
    # if configuring a navmesh, should STATIC MotionType objects be included
    "navmesh_include_static_objects": False,
    # Enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.
    "enable_hbao": False,
    # Amount for ActuationSpec
    "move_actuation_amount": 0.05,
    "turn_actuation_amount": 1,
}
# [/default_sim_settings]

# build SimulatorConfiguration
def make_cfg(settings: Dict[str, Any]):
    r"""Isolates the boilerplate code to create a habitat_sim.Configuration from a settings dictionary.

    :param settings: A dict with pre-defined keys, each a basic simulator initialization parameter.

    Allows configuration of dataset and scene, visual sensor parameters, and basic agent parameters.

    Optionally creates up to one of each of a variety of aligned visual sensors under Agent 0.

    The output can be passed directly into habitat_sim.simulator.Simulator constructor or reconfigure to initialize a Simulator instance.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.frustum_culling = settings.get("frustum_culling", False)
    if "enable_physics" in settings:
        sim_cfg.enable_physics = settings["enable_physics"]
    if "physics_config_file" in settings:
        sim_cfg.physics_config_file = settings["physics_config_file"]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]
    sim_cfg.override_scene_light_defaults = settings.get("override_scene_light_defaults", False)
    sim_cfg.enable_hbao = settings.get("enable_hbao", False)
    sim_cfg.gpu_device_id = 0
    sim_cfg.load_semantic_mesh = True

    if not hasattr(sim_cfg, "scene_id"):
        raise RuntimeError(
            "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
        )
    sim_cfg.scene_id = settings["scene"]

    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensor_specs = []

    def create_camera_spec(**kw_args):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = mn.Vector2i(
            [settings["height"], settings["width"]]
        )
        camera_sensor_spec.position = mn.Vector3(0, settings["sensor_height"], 0)
        for k in kw_args:
            setattr(camera_sensor_spec, k, kw_args[k])
        return camera_sensor_spec

    if settings["color_sensor"]:
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=settings["hfov"],
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            clear_color=settings["clear_color"],
        )
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = create_camera_spec(
            uuid="depth_sensor",
            hfov=settings["hfov"],
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=settings["hfov"],
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(semantic_sensor_spec)

    if settings["ortho_rgba_sensor"]:
        ortho_rgba_sensor_spec = create_camera_spec(
            uuid="ortho_rgba_sensor",
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
            clear_color=settings["clear_color"],
        )
        sensor_specs.append(ortho_rgba_sensor_spec)

    if settings["ortho_depth_sensor"]:
        ortho_depth_sensor_spec = create_camera_spec(
            uuid="ortho_depth_sensor",
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_depth_sensor_spec)

    if settings["ortho_semantic_sensor"]:
        ortho_semantic_sensor_spec = create_camera_spec(
            uuid="ortho_semantic_sensor",
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_semantic_sensor_spec)

    # TODO Figure out how to implement copying of specs
    def create_fisheye_spec(**kw_args):
        fisheye_sensor_spec = habitat_sim.FisheyeSensorDoubleSphereSpec()
        fisheye_sensor_spec.uuid = "fisheye_sensor"
        fisheye_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fisheye_sensor_spec.sensor_model_type = (
            habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
        )

        # The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
        # Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
        # Camera Model, The International Conference on 3D Vision (3DV), 2018
        # You can find the intrinsic parameters for the other lenses in the same table as well.
        fisheye_sensor_spec.xi = -0.27
        fisheye_sensor_spec.alpha = 0.57
        fisheye_sensor_spec.focal_length = [364.84, 364.86]

        fisheye_sensor_spec.resolution = mn.Vector2i(
            [settings["height"], settings["width"]]
        )
        # The default principal_point_offset is the middle of the image
        fisheye_sensor_spec.principal_point_offset = None
        # default: fisheye_sensor_spec.principal_point_offset = [i/2 for i in fisheye_sensor_spec.resolution]
        fisheye_sensor_spec.position = mn.Vector3(0, settings["sensor_height"], 0)
        for k in kw_args:
            setattr(fisheye_sensor_spec, k, kw_args[k])
        return fisheye_sensor_spec

    if settings["fisheye_rgba_sensor"]:
        fisheye_rgba_sensor_spec = create_fisheye_spec(uuid="fisheye_rgba_sensor")
        fisheye_rgba_sensor_spec.clear_color = settings["clear_color"]
        sensor_specs.append(fisheye_rgba_sensor_spec)
    if settings["fisheye_depth_sensor"]:
        fisheye_depth_sensor_spec = create_fisheye_spec(
            uuid="fisheye_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(fisheye_depth_sensor_spec)
    if settings["fisheye_semantic_sensor"]:
        fisheye_semantic_sensor_spec = create_fisheye_spec(
            uuid="fisheye_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(fisheye_semantic_sensor_spec)

    def create_equirect_spec(**kw_args):
        equirect_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        equirect_sensor_spec.uuid = "equirect_rgba_sensor"
        equirect_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        equirect_sensor_spec.resolution = mn.Vector2i(
            [settings["height"], settings["width"]]
        )
        equirect_sensor_spec.position = mn.Vector3(0, settings["sensor_height"], 0)
        for k in kw_args:
            setattr(equirect_sensor_spec, k, kw_args[k])
        return equirect_sensor_spec

    if settings["equirect_rgba_sensor"]:
        equirect_rgba_sensor_spec = create_equirect_spec(uuid="equirect_rgba_sensor")
        equirect_rgba_sensor_spec.clear_color = settings["clear_color"]
        sensor_specs.append(equirect_rgba_sensor_spec)

    if settings["equirect_depth_sensor"]:
        equirect_depth_sensor_spec = create_equirect_spec(
            uuid="equirect_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(equirect_depth_sensor_spec)

    if settings["equirect_semantic_sensor"]:
        equirect_semantic_sensor_spec = create_equirect_spec(
            uuid="equirect_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(equirect_semantic_sensor_spec)

    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = settings["sensor_height"]
    agent_cfg.radius = settings["agent_radius"]
    agent_cfg.sensor_specifications = sensor_specs

    assert type(settings['turn_freq_multiplier']) == int
    assert type(settings['move_freq_multiplier']) == int

    habitat_sim.registry.register_move_fn(
        actions.MoveForwardFreq, name="move_forward_freq", body_action=True
    )
    habitat_sim.registry.register_move_fn(
        actions.TurnLeftFreq, name="turn_left_freq", body_action=True
    )
    habitat_sim.registry.register_move_fn(
        actions.TurnRightFreq, name="turn_right_freq", body_action=True
    )

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings['move_actuation_amount'])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings['turn_actuation_amount'])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings['turn_actuation_amount'])
        ),
        "move_forward_freq": habitat_sim.agent.ActionSpec(
            "move_forward_freq", actions.FreqActuationSpec(
                amount=settings['move_actuation_amount'],
                multiplier=settings['move_freq_multiplier']
            )
        ),
        "turn_left_freq": habitat_sim.agent.ActionSpec(
            "turn_left_freq", actions.FreqActuationSpec(
                amount=settings['turn_actuation_amount'],
                multiplier=settings['turn_freq_multiplier']
            )
        ),
        "turn_right_freq": habitat_sim.agent.ActionSpec(
            "turn_right_freq", actions.FreqActuationSpec(
                amount=settings['turn_actuation_amount'], 
                multiplier=settings['turn_freq_multiplier']
            )
        )
    }

    # construct a NavMeshSettings from default agent paramters for SimulatorConfiguration
    if settings["default_agent_navmesh"]:
        sim_cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
        sim_cfg.navmesh_settings.set_defaults()
        sim_cfg.navmesh_settings.agent_radius = agent_cfg.radius
        sim_cfg.navmesh_settings.agent_height = agent_cfg.height
        sim_cfg.navmesh_settings.cell_size = settings["move_actuation_amount"] / 2
        sim_cfg.navmesh_settings.edge_max_len = agent_cfg.radius * 8 # "A good value for edgeMaxLen is something like agenRadius*8"
        sim_cfg.navmesh_settings.include_static_objects = settings[
            "navmesh_include_static_objects"
        ]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def load_sim_settings(json_path, data_path, dataset=None, scene_name=None):
    with open(json_path, "r") as read_file:
        settings = json.load(read_file)

    if scene_name is None:
        scene_name = settings['scene_name']
    else:
        settings['scene_name'] = scene_name

    if dataset is None:
        dataset = settings['dataset_name']
    else:
        settings['dataset_name'] = dataset

    scene = None

    if dataset == 'gibson':
        scene = os.path.join(
            data_path, f"scene_datasets/gibson/{scene_name}.glb"
        )
        config_path = f"pointnav/gibson/v1/train/content/{scene_name}.json"
    elif dataset == 'replica_cad':
        scene = scene_name

        config_path = f"ReplicaCAD_dataset/replicaCAD.scene_dataset_config.json"

    elif dataset in ['hm3d_minival', 'hm3d_v0.2_minival']:
        scene = os.path.join(
            data_path, f"hm3d_v0.2/minival/{scene_name}/{scene_name.split('-')[-1]}.basis.glb"
        )
        config_path = f"hm3d_v0.2/hm3d_annotated_minival_basis.scene_dataset_config.json"
    # elif dataset == 'mp3d_example':
    #     scene_path = f"scene_datasets/mp3d_example/{scene_name}/{scene_name}.glb"
    #     config_path = None
    # elif dataset == 'test_scenes':
    #     scene_path = f"scene_datasets/habitat-test-scenes/{scene_name}.glb"
    #     config_path = None
    else:
        raise ValueError(f'No such dataset: {dataset}')
    
    settings['scene'] = scene

    if config_path is not None:
        settings['scene_dataset_config_file'] = os.path.join(
            data_path,
            config_path
        )

    return settings


def load_light_settings(json_path):
    with open(json_path, "r") as read_file:
        settings = json.load(read_file)

    return settings
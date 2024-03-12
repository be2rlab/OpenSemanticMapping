import os
import random
import quaternion
import numpy as np

import habitat_sim

from utils.settings import make_cfg
from utils.visual import display_sample


def create_simulator(sim_settings):
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    random.seed(sim_settings["seed"])
    sim.seed(sim_settings["seed"])

    return sim


def get_state_translation_matrix(sensor_state):
    quat = sensor_state.rotation
    trans = sensor_state.position

    rot_mat = quaternion.as_rotation_matrix(quat)
    
    matrix = np.zeros((4, 4), dtype=rot_mat.dtype)
    matrix[:3, :3] = rot_mat
    matrix[:3, 3] = trans
    matrix[3, 3] = 1

    return matrix


def get_camera_matrix(agent):
    sensor_spec = agent.agent_config.sensor_specifications[0]

    w = sensor_spec.resolution[1]
    h = sensor_spec.resolution[0]

    hfov = float(sensor_spec.hfov) * np.pi / 180.

    # hfov = float(agent._sensors['color_sensor'].hfov) * np.pi / 180.
    # fov = float(agent._sensors['color_sensor'].fov) * np.pi / 180.

    f = (1 / np.tan(hfov / 2.)) * w / 2

    K = np.array([
        [f, 0., (h-1) / 2],
        [0., f, (w-1) / 2],
        [0., 0., 1]
    ])

    return K


def do_test_steps(sim, sim_settings, max_frames=5):
    agent = sim.agents[sim_settings['default_agent']]

    action_names = list(agent.agent_config.action_space.keys())

    for _ in range(max_frames):
        action = random.choice(action_names)

        print("action", action)
        observations = sim.step(action)

        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]

        display_sample(rgb, semantic, depth)


def place_agent(sim, sim_settings, start_point, start_rotation=None):
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_point

    if start_rotation is not None:
        agent_state.rotation = start_rotation

    sim.initialize_agent(sim_settings["default_agent"], agent_state)

    return sim
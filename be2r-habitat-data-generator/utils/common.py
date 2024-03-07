import os
import random
import quaternion
import numpy as np

from PIL import Image

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
    w = agent.agent_config.sensor_specifications[0].resolution[0]
    h = agent.agent_config.sensor_specifications[0].resolution[1]

    hfov = float(agent._sensors['color_sensor'].hfov) * np.pi / 180.
    fov = float(agent._sensors['color_sensor'].fov) * np.pi / 180.

    K = np.array([
        [(1 / np.tan(hfov / 2.)) * h / 2 , 0., (h-1) / 2],
        [0., (1 / np.tan(fov / 2.)) * w / 2, (w-1) / 2],
        [0., 0., 1]
    ])

    return K


def save_observed_images(index, observations, depth_scale, image_output_path):
    if "color_sensor" in observations:
        rgb_image = Image.fromarray(observations["color_sensor"]).convert('RGB')

    if "depth_sensor" in observations:
        depth_image = Image.fromarray(observations["depth_sensor"].astype(float) * depth_scale).convert('I')

    if 'semantic_image' in observations:
        semantic_image = Image.fromarray(observations["semantic_sensor"])

    rgb_image.save(os.path.join(image_output_path, 'frame' + str(index).zfill(6) + '.jpg'))
    depth_image.save(os.path.join(image_output_path, 'depth' + str(index).zfill(6) + '.png'))


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
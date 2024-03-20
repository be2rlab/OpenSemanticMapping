import os
import glob
import shutil

import numpy as np

from PIL import Image

from utils.settings import load_sim_settings, load_light_settings


def get_output_paths(dir_path, dataset_name, scene_name):
    output_path = os.path.join(
        dir_path, "generated/", dataset_name, scene_name
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)

    image_output_path = os.path.join(output_path, 'results/')

    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    return output_path, image_output_path


class ExperimentLogger:
    def __init__(self, sim_settings_path, light_settings_path, package_dir_path='./') -> None:
        self.sim_settings = load_sim_settings(sim_settings_path, os.path.join(package_dir_path, '../data/'))
        self.light_settings = load_light_settings(light_settings_path)

        self.output_dir_path, self.image_output_path = get_output_paths(
            package_dir_path, 
            self.sim_settings['dataset_name'], 
            self.sim_settings['scene_name']
        )

        shutil.copy(sim_settings_path, os.path.join(self.output_dir_path, 'sim_settings.json'))
        shutil.copy(light_settings_path, os.path.join(self.output_dir_path, 'light_settings.json'))

        self.depth_scale = self.sim_settings['depth_scale']

        self.__step_index = 0

    def save_step(self, observations, transformation_matrix):
        if "color_sensor" in observations:
            rgb_image = Image.fromarray(observations["color_sensor"]).convert('RGB')
            rgb_image.save(os.path.join(
                self.image_output_path, 
                f'frame{self.get_str_index()}.jpg'
            ))

        if "depth_sensor" in observations:
            depth_image = Image.fromarray(
                observations["depth_sensor"].astype(float) * self.depth_scale
            ).convert('I')

            depth_image.save(os.path.join(
                self.image_output_path, 
                f'depth{self.get_str_index()}.png'
            ))

        if 'semantic_image' in observations:
            semantic_image = Image.fromarray(observations["semantic_sensor"])
            # depth_image.save(os.path.join(self.image_output_path, f'semantic{self.get_str_index()}.png'))

        with open(os.path.join(self.output_dir_path, "traj.txt"), "a") as traj_file:
            np.savetxt(traj_file, transformation_matrix, newline=" ")
            traj_file.write("\n")

        self.__step_index += 1

    def add_entry(self, message):
        with open(os.path.join(self.output_dir_path, "log.txt"), "a") as log_file:
            log_file.write(f'[{self.get_str_index()}] {message}\n')

    def get_step_index(self):
        return self.__step_index
    
    def get_str_index(self):
        return str(self.__step_index).zfill(6)

    def get_settings(self):
        return self.sim_settings, self.light_settings
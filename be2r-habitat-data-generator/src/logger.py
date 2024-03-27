import os
import glob
import shutil

import numpy as np

from PIL import Image

from utils.settings import load_sim_settings, load_light_settings
from utils.visual import display_sample


def get_output_paths(dir_path, dataset_name, scene_name, label):
    output_path = os.path.join(
        dir_path, "generated/", dataset_name, scene_name, label
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)

    image_output_path = os.path.join(output_path, 'results/')

    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    sim_settings_output_path = os.path.join(output_path, 'configs/sim_settings/')
    if not os.path.exists(sim_settings_output_path):
        os.makedirs(sim_settings_output_path)

    lights_output_path = os.path.join(output_path, 'configs/lighting/')
    if not os.path.exists(lights_output_path):
        os.makedirs(lights_output_path)


    return output_path, image_output_path, sim_settings_output_path, lights_output_path


class ExperimentLogger:
    def __init__(self, sim_settings_filename, package_dir_path='./', display=False, display_freq=50) -> None:
        self.__sim_settings_filename = sim_settings_filename

        configs_path = os.path.join(package_dir_path, 'configs/')
        sim_settings_path = os.path.join(configs_path, 'sim_settings/', sim_settings_filename)

        self.sim_settings = load_sim_settings(sim_settings_path, os.path.join(package_dir_path, '../data/'))

        light_settings_path = os.path.join(configs_path, 'lighting/', self.sim_settings['light_settings_filename'])
        self.light_settings = load_light_settings(light_settings_path)

        self.output_dir_path, self.image_output_path, sim_settings_output_path, lights_output_path = get_output_paths(
            package_dir_path, 
            self.sim_settings['dataset_name'], 
            self.sim_settings['scene_name'],
            self.sim_settings['label']
        )

        shutil.copy(sim_settings_path, sim_settings_output_path)
        shutil.copy(light_settings_path, lights_output_path)

        self.depth_scale = self.sim_settings['depth_scale']

        self._step_index = 0

        self._display = display
        self._display_freq = display_freq

    def __str__(self) -> str:
        return self.__sim_settings_filename
    
    def set_display(self, display: bool) -> None:
        self._display = display

    def save_step(self, observations, transformation_matrix, display=None):
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

        if 'semantic_sensor' in observations:
            semantic_image = Image.fromarray(
                observations["semantic_sensor"]
            ).convert('I')

            semantic_image.save(os.path.join(
                self.image_output_path, 
                f'semantic{self.get_str_index()}.png'
            ))

        with open(os.path.join(self.output_dir_path, "traj.txt"), "a") as traj_file:
            np.savetxt(traj_file, transformation_matrix, newline=" ")
            traj_file.write("\n")

        display = self._display if display is None else display

        if display:
            if self._step_index % self._display_freq == 0:
                rgb = observations["color_sensor"]
                semantic = observations.get("semantic_sensor", None)
                depth = observations.get("depth_sensor", None)

                display_sample(rgb, semantic_obs=semantic, depth_obs=depth)

        self._step_index += 1

    def save_camera_params(self, camera_matrix, printed=None):
        printed = self._display if printed is None else printed

        message = f"camera_matrix = \n{camera_matrix}\n\ndepth_scale = {self.sim_settings['depth_scale']}"

        if printed:
            print(message)

        with open(os.path.join(self.output_dir_path, "camera_params.txt"), "w") as file:
            file.write(message)

    def add_entry(self, message, printed=None):
        printed = self._display if printed is None else printed

        if printed:
            print(message)

        with open(os.path.join(self.output_dir_path, "log.txt"), "a") as log_file:
            log_file.write(f'[{self.get_str_index()}] {message}\n')

    def get_step_index(self):
        return self._step_index
    
    def get_str_index(self):
        return str(self._step_index).zfill(6)

    def get_settings(self):
        return self.sim_settings, self.light_settings
    
    def get_settings_name(self):
        return self.__sim_settings_filename
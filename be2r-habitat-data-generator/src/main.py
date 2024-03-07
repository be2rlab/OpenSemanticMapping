import numpy as np

from tqdm import tqdm

import habitat_sim
from habitat.utils.visualizations import maps

from src.lights import change_lights
from utils.visual import convert_points_to_topdown, display_map, display_sample
from utils.common import get_state_translation_matrix, save_observed_images


def get_navigable_point(sim, sim_settings):
    finder = sim.pathfinder
    radius = sim_settings['agent_radius']

    point = finder.get_random_navigable_point(max_tries=1000)

    while np.isnan(point).sum() > 0 or finder.distance_to_closest_obstacle(point, radius * 2) < radius:
        point = finder.get_random_navigable_point(max_tries=1000)

    return point


def generate_scenario(sim, nav_points_count, sim_settings, display=True):
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        print("NavMesh area = " + str(sim.pathfinder.navigable_area))
        print("Bounds = ", *sim.pathfinder.get_bounds())

        pathfinder_seed = sim_settings['seed']
        sim.pathfinder.seed(pathfinder_seed)
        navigatable_points = [get_navigable_point(sim, sim_settings) for _ in range(nav_points_count)]
        print("Random navigable points : ", *navigatable_points)
        print("Are points navigable? " + str(all([sim.pathfinder.is_navigable(nav_points, sim_settings['agent_radius']) for nav_points in navigatable_points])))

        start_point = get_navigable_point(sim, sim_settings)
        
        print("Start point : " + str(start_point))
        vis_points = [start_point] + navigatable_points

        if display:
            meters_per_pixel = 0.1

            xy_vis_points = convert_points_to_topdown(
                sim.pathfinder, vis_points, meters_per_pixel
            )

            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )

            full_top_down_map = None

            for point in vis_points:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height=point[1], meters_per_pixel=meters_per_pixel
                )

                if full_top_down_map is None:
                    full_top_down_map = top_down_map
                else:
                    full_top_down_map = np.maximum(top_down_map, full_top_down_map)
                
            full_top_down_map = recolor_map[full_top_down_map]

            display_map(full_top_down_map, agent_position=xy_vis_points[0], key_points=xy_vis_points[1:])

    return start_point, navigatable_points


def run_scenario(sim, sim_settings, light_settings, start_point, navigatable_points, depth_scale, image_output_path, display=True, display_freq=50):
    trajectory = []
    step_index = 0
          
    for nav_index, nav_point in enumerate(tqdm(navigatable_points)):
        sim = change_lights(sim, sim_settings, light_settings, nav_index)
        agent = sim.agents[sim_settings["default_agent"]]
        follower = sim.make_greedy_follower(sim_settings["default_agent"])

        try:
            path = follower.find_path(nav_point)
        except Exception as e:
            print('-' * 20, f'Path exception: {e}. Skip goal.', '-' * 20, sep='\n')
            continue

        with tqdm(total=len(path)) as pbar:
            while True:
                try:
                    action = follower.next_action_along(nav_point)
                except Exception as e:
                    print('-' * 20, f'Exception: {e}. Skip goal.', '-' * 20, sep='\n')
                    break

                if action in ['error', 'stop', None]:
                    print("Final action: ", action, sep='')
                    break

                observations = sim.step(action)

                trajectory.append(get_state_translation_matrix(agent.state.sensor_states['color_sensor']).flatten())
                save_observed_images(step_index, observations, depth_scale, image_output_path)

                if display:
                    # print("action", action)

                    if step_index % display_freq == 0:
                        display_sample(observations['color_sensor'], observations['semantic_sensor'], observations['depth_sensor'])

                step_index += 1
                pbar.update()

    return trajectory
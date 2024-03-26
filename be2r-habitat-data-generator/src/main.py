import numpy as np

try:
    __IPYTHON__
    from tqdm.notebook import tqdm, trange
except NameError:
    from tqdm import tqdm, trange

import habitat_sim
from habitat.utils.visualizations import maps

from src.lights import change_lights
from utils.visual import convert_points_to_topdown, display_map, display_sample
from utils.common import get_state_transform_matrix


def check_path_finding(follower, point):
    try:
        follower.find_path(point)
    except Exception:
        return False
    
    return True


def get_navigable_point(sim, sim_settings, prev_point=None):
    finder = sim.pathfinder
    radius = sim_settings['agent_radius']

    if prev_point is None:
        get_point = lambda: finder.get_random_navigable_point(max_tries=1000)
    else:
        get_point = lambda: finder.get_random_navigable_point_near(
            circle_center=prev_point,
            radius = 1_000 * sim_settings['move_actuation_amount'],
            max_tries=1000,
        )

    point = get_point()

    while np.isnan(point).sum() > 0 or  \
            finder.distance_to_closest_obstacle(point, radius * 2) < radius:
        point = get_point()

    return point


def generate_scenario(sim, sim_settings, display=True):
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        nav_points_number= sim_settings['nav_points_number']

        pathfinder_seed = sim_settings['seed']
        sim.pathfinder.seed(pathfinder_seed)

        start_point = get_navigable_point(sim, sim_settings)

        vis_points = [start_point]

        for _ in trange(nav_points_number):
            vis_points.append(get_navigable_point(sim, sim_settings, vis_points[-1]))

        navigatable_points = vis_points[1:]

        if display:
            print("NavMesh area = " + str(sim.pathfinder.navigable_area))
            print("Bounds = ", *sim.pathfinder.get_bounds())
            print("Start point : " + str(start_point))
            print("Random navigable points : ", *navigatable_points)
            print(
                "Are points navigable? " + 
                str(all([
                    sim.pathfinder.is_navigable(nav_points, sim_settings['move_actuation_amount']) for nav_points in navigatable_points
                ]))
            )

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


def run_scenario(sim, sim_settings, light_settings, navigatable_points, logger, display=True, display_freq=50, bar_inds=[0, 1]):
    for nav_index, nav_point in enumerate(tqdm(navigatable_points, desc=f'{logger} proccessing', position=bar_inds[0])):
        sim, light_setup = change_lights(sim, sim_settings, light_settings, nav_index)

        msg = f'Light setup: {light_setup}'
        logger.add_entry(msg, printed=display)

        agent = sim.agents[sim_settings["default_agent"]]
        follower = sim.make_greedy_follower(sim_settings["default_agent"])

        try:
            path = follower.find_path(nav_point)
        except Exception as e:
            msg = f'Path exception: {e}. Skip goal.'

            logger.add_entry(msg, printed=display)
            continue

        pbar_kwargs = {
            'total': len(path),
            'desc': f'Navigation #{nav_index}',
            'position': bar_inds[1] if len(bar_inds) >= 2 else None,
            'disable': len(bar_inds) < 2,
            'leave': False
        }

        with tqdm(**pbar_kwargs) as pbar:
            while True:
                pbar.update()

                try:
                    action = follower.next_action_along(nav_point)
                except Exception as e:
                    msg = f'Path exception: {e}. Skip goal.'

                    logger.add_entry(msg, printed=display)
                    break

                if action in ['error', 'stop', None]:
                    msg = f"Final action: {action}"
                    logger.add_entry(msg)
                    break
                else:
                    logger.add_entry(f'Action: {action}')

                observations = sim.step(action)

                if display:
                    if logger.get_step_index() % display_freq == 0:
                        rgb = observations["color_sensor"]
                        semantic = observations.get("semantic_sensor", None)
                        depth = observations.get("depth_sensor", None)

                        display_sample(rgb, semantic_obs=semantic, depth_obs=depth)

                logger.save_step(
                    observations,
                    get_state_transform_matrix(agent.state.sensor_states['color_sensor']).flatten()
                )


def run_scenario_optimized(sim_planner, sim_agent, sim_settings_planner,
                      sim_settings_agent, light_settings, navigatable_points,
                      logger, step_multiplier, angle_multiplier,
                      display=True, display_freq=50, bar_inds=[0, 1]):
    
    for nav_index, nav_point in enumerate(tqdm(navigatable_points, desc=f'{logger} proccessing', position=bar_inds[0])):
        
        sim_planner, light_setup_planner = change_lights(sim_planner, sim_settings_planner, light_settings, nav_index)
        sim_agent, light_setup_agent = change_lights(sim_agent, sim_settings_agent, light_settings, nav_index)

        msg = f'Light setup: {light_setup_agent}'

        logger.add_entry(msg, printed=display)

        agent_agent = sim_agent.agents[sim_settings_agent["default_agent"]]
        agent_planner = sim_planner.agents[sim_settings_planner["default_agent"]]
        
        print(f"agent: {sim_settings_planner['default_agent']}")
        
        follower = sim_planner.make_greedy_follower(sim_settings_planner["default_agent"])
        
        # 1 поворот в планнере равен 10 поворотам агента
        # 1 шаг в планнере равен 50 шагам агента
        try:
            path = follower.find_path(nav_point)
        except Exception as e:
            msg = f'Path exception: {e}. Skip goal.'

            logger.add_entry(msg, printed=display)
            continue
        
        print(f"Nav index: {nav_index}")
        print(f"Nav point: {nav_point}")
        agent_path = transform(path, step_multiplier, angle_multiplier)
        print(f"Planner Path: {path}")
        print(f"Agent Path: {agent_path}")
        
                
        
        with tqdm(total=len(agent_path), desc=f'Navigation #{nav_index}', position=bar_inds[1]) as pbar:
            for action in agent_path:
                observations = sim_agent.step(action)
                
                if display:
                    if logger.get_step_index() % display_freq == 0:
                        rgb = observations["color_sensor"]
                        semantic = observations.get("semantic_sensor", None)
                        depth = observations.get("depth_sensor", None)

                        display_sample(rgb, semantic_obs=semantic, depth_obs=depth)

                logger.save_step(
                    observations,
                    get_state_transform_matrix(agent_agent.state.sensor_states['color_sensor']).flatten()
                )
                pbar.update()

def transform(old, step_multiplier, angle_multiplier):
    res = []
    
    for elem in old:
        if elem == 'turn_left':
            for _ in range(angle_multiplier):
                res.append(elem)
        elif elem == 'turn_right':
            for _ in range(angle_multiplier):
                res.append(elem)
        elif elem == 'move_forward':
            for _ in range(step_multiplier):
                res.append(elem)
        elif elem in ['error', 'stop', None]:
            res.append(elem)

    return res




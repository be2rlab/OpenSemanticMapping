import numpy as np

from tqdm.auto import tqdm, trange

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

    get_point = lambda: finder.get_random_navigable_point(max_tries=1_000)

    point = get_point()

    while np.isnan(point).sum() > 0:
        point = get_point()

    return point


def generate_scenario(sim, sim_settings, display=True):
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        nav_points_number = sim_settings['nav_points_number']

        sim.pathfinder.seed(sim_settings['seed'])

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


def run_scenario(sim, sim_settings, light_settings, navigatable_points, logger, display=True, bar_inds=[0, 1], frequent=True):
    for nav_index, nav_point in enumerate(tqdm(navigatable_points, desc=f'{logger} proccessing', position=bar_inds[0])):
        sim, light_setup = change_lights(sim, sim_settings, light_settings, nav_index)

        msg = f'Light setup: {light_setup}'
        logger.add_entry(msg, printed=display)

        follower = sim.make_greedy_follower(
            agent_id = sim_settings['default_agent'],
            goal_radius = sim_settings.get('goal_radius', None),
            forward_key = 'move_forward',
            left_key = 'turn_left',
            right_key = 'turn_right'
        )

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
            _navigation_loop(sim, sim_settings, follower, nav_point, pbar, logger, display, frequent)


def _navigation_loop(sim, sim_settings, follower, nav_point, pbar, logger, display, frequent):
    agent = sim.agents[sim_settings["default_agent"]]

    logger.add_entry('Start navigating.', printed=display)

    while True:
        pbar.update()

        try:
            action = follower.next_action_along(nav_point)
        except Exception as e:
            msg = f'Path exception: {e}. Reached or error occured. Skip goal.'

            logger.add_entry(msg, printed=display)
            break

        if action in ['error', 'stop', None]:
            msg = f"Final action: {action}"
            logger.add_entry(msg, printed=False)
            break
        else:
            logger.add_entry(f'Action: {action}', printed=False)

        if frequent:
            _navigation_step_freq(sim, sim_settings, agent, action, logger)
        else:
            _navigation_step(sim, agent, action, logger)

    logger.add_entry(f'Stop navigating.', printed=display)


def _navigation_step_freq(sim, sim_settings, agent, action, logger):
    multiplier = {
        'move_forward': sim_settings['move_freq_multiplier'],
        'turn_left': sim_settings['turn_freq_multiplier'],
        'turn_right': sim_settings['turn_freq_multiplier'],
    }[action]

    for _ in range(multiplier):
        observations = sim.step(action + '_freq')

        logger.save_step(
            observations,
            get_state_transform_matrix(agent.state.sensor_states['color_sensor']).flatten()
        )


def _navigation_step(sim, agent, action, logger):
    observations = sim.step(action)

    logger.save_step(
        observations,
        get_state_transform_matrix(agent.state.sensor_states['color_sensor']).flatten()
    )
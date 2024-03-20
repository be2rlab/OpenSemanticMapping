import numpy as np

from habitat_sim.gfx import LightInfo, LightPositionModel

from utils.settings import make_cfg


default_light_settings = {
    'lights': {
        '0': {
            'position': [0., 0., 0., 1.0],
            'intensity': 10.0,
            'color': [1.0, .0, .0],
            'type': 'camera',
            'life_intervals': [
                [None, 1]
            ]
        },
        '1': {
            'position': [0., 0., 0., 1.0],
            'intensity': 10.0,
            'color': [.0, 1.0, .0],
            'type': 'camera',
            'life_intervals': [
                [1, None]
            ]
        },
    }
}


light_types_mapping = {
    'camera': LightPositionModel.Camera,
    'point': LightPositionModel.Global,
    'directional': LightPositionModel.Object,
}


def set_light_setup(sim, settings, life_index):
    setup = []
    setup_name = f'lights_setup_{life_index}'

    for setup_id, setup_params in settings['lights'].items():
        intervals = setup_params['life_intervals']

        for interval in intervals:
            if (interval[0] is None or interval[0] <= life_index) and (interval[1] is None or life_index < interval[1]):
                intens_koef = 0.1 if setup_params['intensity'] < 0 else 2.5
                info = {
                    'model': light_types_mapping[setup_params['type']],
                    'vector': setup_params['position'],
                    'color': list(np.clip(setup_params['color'], 0.0, 1.0) * setup_params['intensity'] * intens_koef)
                }

                if len(info['vector']) == 3:
                    info['vector'] += [1.0]

                setup.append((setup_id, info))

                break

    sim.set_light_setup(
        [LightInfo(**info) for (_, info) in setup], 
        setup_name
    )

    return sim, setup_name, setup


def change_lights(sim, sim_settings, light_settings, life_index):
    if not sim_settings['override_scene_light_defaults']:
        return sim, None
    
    sim, setup_name, setup = set_light_setup(sim, light_settings, life_index)

    sim_settings = sim_settings.copy()
    sim_settings['scene_light_setup'] = setup_name
    sim_settings['override_scene_light_defaults'] = True

    new_cfg = make_cfg(sim_settings)

    agent_states = [agent.get_state() for agent in sim.agents]

    sim.reconfigure(new_cfg)

    for i, state in enumerate(agent_states):
        sim.initialize_agent(i, state)

    return sim, setup
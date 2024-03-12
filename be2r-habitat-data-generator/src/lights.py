from habitat_sim.gfx import LightInfo, LightPositionModel

from utils.settings import make_cfg


default_light_settings = {
    'lights': [
        {
            'info': {
                'vector': [0., 0., 0., 1.0],
                'color': [10.0, .0, .0],
                'model': 'camera',
            },
            'life_interval': [None, 1]    
        },
        {
            'info': {
                'vector': [0., 0., 0., 1.0],
                'color': [.0, 10.0, .0],
                'model': 'camera',
            },
            'life_interval': [1, None]    
        }
    ]
}


light_models_mapping = {
    'camera': LightPositionModel.Camera,
    'global': LightPositionModel.Global,
    'object': LightPositionModel.Object,
}


def set_light_setup(sim, settings, life_index):
    setup = []
    setup_name = f'lights_setup_{life_index}'

    for setup_params in settings['lights']:
        interval = setup_params['life_interval']

        if (interval[0] is None or interval[0] <= life_index) and (interval[1] is None or life_index < interval[1]):
            info = setup_params['info'].copy()
            info['model'] = light_models_mapping[info['model']]
            setup.append(info)

    sim.set_light_setup(
        [LightInfo(**info) for info in setup], 
        setup_name
    )

    return sim, setup_name, setup


def change_lights(sim, sim_settings, light_settings, life_index):
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
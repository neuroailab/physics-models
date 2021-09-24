import os
from physion.data.config import get_cfg_defaults

def build_paths(name, scenarios, filepattern, traindir, testdir):
    return {
        'name': name, 
        'train': [os.path.join(traindir, scenario, filepattern) for scenario in scenarios], 
        'test': [os.path.join(testdir, scenario, filepattern) for scenario in scenarios],
        } 

def get_config(cfg_file):
    cfg = get_cfg_defaults()
    if not os.path.isabs(cfg_file): # if not absolute path, looks in this file's dir (i.e. `physion/data/`)
        dirname =  os.path.dirname(__file__)
        cfg_file = os.path.join(dirname, cfg_file)
    cfg.merge_from_file(cfg_file)
    for dst_attr, src_attr in [('READOUT_FILE_PATTERN', 'PRETRAINING_FILE_PATTERN'), ('READOUT_SCENARIOS', 'PRETRAINING_SCENARIOS')]: # copy readout settings from pretraining if not set
        if getattr(cfg, dst_attr) is None:
            setattr(cfg, dst_attr, getattr(cfg, src_attr))
    cfg.freeze()
    return cfg

def add_seed_to_data_spaces(seeds, data_spaces):
    full_data_spaces = [] # full data space with seed
    for seed in seeds:
        for space in data_spaces:
            space = space.copy()
            space['seed'] = seed
            full_data_spaces.append(space)
    return full_data_spaces

def get_data_spaces(cfg_file):
    cfg = get_config(cfg_file)

    data_spaces = [] # only pretraining and readout spaces, without seed
    for scenario in cfg.PRETRAINING_SCENARIOS:
        if 'only' in cfg.PRETRAINING_PROTOCOLS:
            if cfg.READOUT_PROTOCOL == 'full':
                readout_scenarios = cfg.READOUT_SCENARIOS
            else:
                assert scenario in cfg.READOUT_SCENARIOS, '{} not in {}, but using "{}" readout protocol'.format(scenario, cfg.READOUT_SCENARIOS, cfg.READOUT_PROTOCOL)
                readout_scenarios = [scenaro]
            space = {
                'pretraining': build_paths(scenario, [scenario], cfg.PRETRAINING_FILE_PATTERN, cfg.PRETRAINING_TRAIN_DIR, cfg.PRETRAINING_TEST_DIR),
                'readout': [build_paths(scenario, [scenario], cfg.READOUT_FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR) for scenario in readout_scenarios],
                }
            data_spaces.append(space)

        if 'abo' in cfg.PRETRAINING_PROTOCOLS:
            assert len(cfg.PRETRAINING_SCENARIOS) > 1, 'Must have more than one scenario to do all-but-one protocol.'
            abo_scenarios = [s for s in cfg.PRETRAINING_SCENARIOS if s is not scenario]
            if cfg.READOUT_PROTOCOL == 'full':
                readout_scenarios = cfg.READOUT_SCENARIOS
            else:
                assert scenario in cfg.READOUT_SCENARIOS, '{} not in {}, but using "{}" readout protocol'.format(scenario, cfg.READOUT_SCENARIOS, cfg.READOUT_PROTOCOL)
                readout_scenarios = [scenaro]
            space = {
                'pretraining': build_paths('no_'+scenario, abo_scenarios, cfg.PRETRAINING_FILE_PATTERN, cfg.PRETRAINING_TRAIN_DIR, cfg.PRETRAINING_TEST_DIR), # train on all but the scenario
                'readout': [build_paths(scenario, [scenario], cfg.READOUT_FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR) for scenario in readout_scenarios],
                }
            data_spaces.append(space)
        
    if 'all' in cfg.PRETRAINING_PROTOCOLS:
        assert len(cfg.PRETRAINING_SCENARIOS) > 1, 'Must have more than one scenario to do all protocol.'
        space = {
            'pretraining': build_paths('all', cfg.PRETRAINING_SCENARIOS, cfg.PRETRAINING_FILE_PATTERN, cfg.PRETRAINING_TRAIN_DIR, cfg.PRETRAINING_TEST_DIR), # train on all scenarios
            'readout': [build_paths(scenario, [scenario], cfg.READOUT_FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR) for scenario in cfg.READOUT_SCENARIOS], # readout on each scenario individually
            }
        data_spaces.append(space)

    if isinstance(cfg.SEEDS, list):
        seeds = cfg.SEEDS
    else:
        assert isinstance(cfg.SEEDS, int)
        seeds = list(range(cfg.SEEDS))
    full_data_spaces = add_seed_to_data_spaces(seeds, data_spaces)

    return full_data_spaces

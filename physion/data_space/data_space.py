import os
import socket
import yaml

DEFAULTS = {
    'suffix': '',
}

class DataManager():
    def __init__(self, data_settings):
        self.data_settings = data_settings

    def get_setting(self, setting, mode, phase):
        try:
            val = self.data_settings[phase][mode][setting]
        except KeyError:
            try:
                val = self.data_settings[phase][setting]
            except KeyError:
                try:
                    val = self.data_settings[setting]
                except KeyError:
                    if setting in DEFAULTS:
                        val = DEFAULTS[setting]
                    else:
                        print(self.data_settings)
                        print(f'{setting} not found for phase {phase} and mode {mode}')
                        raise
        return val

    def get_scenarios(self, mode, phase):
        scenarios = self.get_setting('scenarios', mode, phase)
        suffix = self.get_setting('suffix', mode, phase)
        assert isinstance(scenarios, list), f'Scenarios for {phase} {mode} has type {type(scenarios)}, expected list'
        return [scenario + suffix for scenario in scenarios]

    def get_data_dir(self, mode, phase):
        data_dir = self.get_setting('dir', mode, phase)
        if not os.path.isabs(data_dir):
            data_dir = self.add_basedir(data_dir)
        return data_dir

    def get_file_pattern(self, mode, phase):
        return self.get_setting('file_pattern', mode, phase)

    @staticmethod
    def add_basedir(reldir):
        dirname =  os.path.dirname(__file__)
        hostname = socket.gethostname()
        basedir_file = os.path.join(dirname, 'basedir.yaml')
        if os.path.isfile(basedir_file):
            basedir_dict = yaml.safe_load(open(basedir_file, 'rb'))
            assert hostname in basedir_dict, f'{hostname} not found in {basedir_file}'
            basedir = basedir_dict.get(hostname)
        return os.path.join(basedir, reldir)

    def build_paths(self, name, scenarios, phase):
        res = {
            'name': name, 
            } 
        for mode in ['train', 'test']:
            curr_scenarios = scenarios[mode]
            if not isinstance(curr_scenarios, list):
                assert isinstance(curr_scenarios, str)
                curr_scenarios = [curr_scenarios]
            data_dir = self.get_data_dir(mode, phase)
            file_pattern = self.get_file_pattern(mode, phase)
            res[mode] = [os.path.join(data_dir, scenario, file_pattern) for scenario in curr_scenarios]
        return res

def get_readout_paths(data_manager, readout_protocol, pretraining_train_scenario):
    readout_train_scenarios = data_manager.get_scenarios('train', 'readout')
    readout_test_scenarios = data_manager.get_scenarios('test', 'readout')
    if readout_protocol == 'full':
        readout_paths = [data_manager.build_paths(readout_train_scenario, {'train': readout_train_scenario, 'test': readout_test_scenario}, 'readout') for readout_train_scenario, readout_test_scenario in zip(readout_train_scenarios, readout_test_scenarios)]
    else: # "minimal" protocal matches readout train scenario to pretraining train scenario
        pretrain_train_wo_suffix = pretraining_train_scenario.replace(data_manager.get_setting('suffix', 'train', 'pretraining'), '', 1)
        assert pretrain_train_wo_suffix in readout_train_scenarios, '{} not in {}, but using "{}" readout protocol'.format(pretrain_train_wo_suffix, readout_train_scenarios, readout_protocol)
        readout_train_scenario = pretrain_train_wo_suffix
        readout_test_scenario = readout_test_scenarios[readout_train_scenarios.index(pretrain_train_wo_suffix)]
        readout_paths = [data_manager.build_paths(readout_train_scenario, {'train': readout_train_scenario, 'test': readout_test_scenario}, 'readout')]
    return readout_paths

def get_only_space(
    data_manager,
    readout_protocol,
    ):
    data_spaces = []
    pretraining_train_scenarios = data_manager.get_scenarios('train', 'pretraining')
    pretraining_test_scenarios = data_manager.get_scenarios('test', 'pretraining')
    for pretraining_train_scenario, pretraining_test_scenario in zip(pretraining_train_scenarios, pretraining_test_scenarios):
        space = {
            'pretraining': data_manager.build_paths(pretraining_train_scenario, {'train': pretraining_train_scenario, 'test': pretraining_test_scenario}, 'pretraining'),
            'readout': get_readout_paths(data_manager, readout_protocol, pretraining_train_scenario),
            }
        data_spaces.append(space)
    return data_spaces

def get_abo_space(
    data_manager,
    readout_protocol,
    ):
    data_spaces = []
    pretraining_train_scenarios = data_manager.get_scenarios('train', 'pretraining')
    pretraining_test_scenarios = data_manager.get_scenarios('test', 'pretraining')
    readout_train_scenarios = data_manager.get_scenarios('train', 'readout')
    readout_test_scenarios = data_manager.get_scenarios('test', 'readout')
    assert len(pretraining_train_scenarios) > 1, 'Must have more than one scenario to do all-but-one pretraining protocol.' # just check train since train and test should be same length
    for pretraining_train_scenario, pretraining_test_scenario in zip(pretraining_train_scenarios, pretraining_test_scenarios):
        # build abo scenarios
        abo_pretraining_scenarios = list(zip(pretraining_train_scenarios, pretraining_test_scenarios))
        abo_pretraining_scenarios.remove((pretraining_train_scenario, pretraining_test_scenario))
        abo_pretraining_train_scenarios, abo_pretraining_test_scenarios = [list(t) for t in zip(*abo_pretraining_scenarios)]

        space = {
            'pretraining': data_manager.build_paths('no_'+pretraining_train_scenario, {'train': abo_pretraining_train_scenarios, 'test': abo_pretraining_test_scenarios}, 'pretraining'),
            'readout': get_readout_paths(data_manager, readout_protocol, pretraining_train_scenario),
            }
        data_spaces.append(space)
    return data_spaces

def get_all_space(
    data_manager,
    ):
    pretraining_train_scenarios = data_manager.get_scenarios('train', 'pretraining')
    pretraining_test_scenarios = data_manager.get_scenarios('test', 'pretraining')
    readout_train_scenarios = data_manager.get_scenarios('train', 'readout')
    readout_test_scenarios = data_manager.get_scenarios('test', 'readout')
    assert len(pretraining_train_scenarios) > 1, f'Must have more than one scenario to do all pretraining protocol.' # just check train since train and test should be same length
    pretraining_train_suffix = data_manager.get_setting('suffix', 'train', 'pretraining')
    space = {
        'pretraining': data_manager.build_paths('all'+pretraining_train_suffix, {'train': pretraining_train_scenarios, 'test': pretraining_test_scenarios}, 'pretraining'),
        'readout': [data_manager.build_paths(readout_train_scenario, {'train': readout_train_scenario, 'test': readout_test_scenario}, 'readout') for readout_train_scenario, readout_test_scenario in zip(readout_train_scenarios, readout_test_scenarios)]
        }
    data_spaces = [space]
    return data_spaces

def get_data_spaces(
    pretraining_protocols=('all', 'abo', 'only'),
    readout_protocol='minimal', # {'full'|'minimal'}: 'minimal' only does readout on matching scenario to pretraining
    **data_settings
    ):
    data_manager = DataManager(data_settings)

    data_spaces = [] # only pretraining and readout spaces, without seed
    if 'only' in pretraining_protocols:
        data_spaces.extend(get_only_space(
            data_manager,
            readout_protocol,
            ))
    if 'abo' in pretraining_protocols:
        data_spaces.extend(get_abo_space(
            data_manager,
            readout_protocol,
            ))
    if 'all' in pretraining_protocols:
        data_spaces.extend(get_all_space(
            data_manager,
            ))
    # print(*data_spaces, sep='\n')
    return data_spaces

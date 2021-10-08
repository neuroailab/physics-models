import os
import socket
import yaml

def build_paths(name, scenarios, filepattern, traindir, testdir):
    dirname =  os.path.dirname(__file__)
    hostname = socket.gethostname()
    basedir_file = os.path.join(dirname, 'basedir.yaml')
    if os.path.isfile(basedir_file):
        basedir_dict = yaml.safe_load(open(basedir_file, 'rb'))
        basedir = basedir_dict.get(hostname)
    if basedir is not None:
        if not os.path.isabs(traindir):
            traindir = os.path.join(basedir, traindir)
        if not os.path.isabs(testdir):
            testdir = os.path.join(basedir, testdir)
    return {
        'name': name, 
        'train': [os.path.join(traindir, scenario, filepattern) for scenario in scenarios], 
        'test': [os.path.join(testdir, scenario, filepattern) for scenario in scenarios],
        } 

def get_data_spaces(
    pretraining_train_dir,
    pretraining_test_dir,
    pretraining_scenarios,
    readout_train_dir,
    readout_test_dir,
    pretraining_file_pattern='*.hdf5',
    pretraining_protocols=('all', 'abo', 'only'),
    readout_scenarios=None,
    readout_file_pattern=None,
    readout_protocol='minimal', # {'full'|'minimal'}: 'minimal' only does readout on matching scenario to pretraining
    ):
    if readout_file_pattern is None:
        readout_file_pattern = pretraining_file_pattern
    if readout_scenarios is None:
        readout_scenarios = pretraining_scenarios

    data_spaces = [] # only pretraining and readout spaces, without seed
    for scenario in pretraining_scenarios:
        if 'only' in pretraining_protocols:
            if readout_protocol == 'full':
                readout_scenarios = readout_scenarios
            else:
                assert scenario in readout_scenarios, '{} not in {}, but using "{}" readout protocol'.format(scenario, readout_scenarios, readout_protocol)
                readout_scenarios = [scenario]
            space = {
                'pretraining': build_paths(scenario, [scenario], pretraining_file_pattern, pretraining_train_dir, pretraining_test_dir),
                'readout': [build_paths(scenario, [scenario], readout_file_pattern, readout_train_dir, readout_test_dir) for scenario in readout_scenarios],
                }
            data_spaces.append(space)

        if 'abo' in pretraining_protocols:
            assert len(pretraining_scenarios) > 1, 'Must have more than one scenario to do all-but-one protocol.'
            abo_scenarios = [s for s in pretraining_scenarios if s is not scenario]
            if readout_protocol == 'full':
                readout_scenarios = readout_scenarios
            else:
                assert scenario in readout_scenarios, '{} not in {}, but using "{}" readout protocol'.format(scenario, readout_scenarios, readout_protocol)
                readout_scenarios = [scenario]
            space = {
                'pretraining': build_paths('no_'+scenario, abo_scenarios, pretraining_file_pattern, pretraining_train_dir, pretraining_test_dir), # train on all but the scenario
                'readout': [build_paths(scenario, [scenario], readout_file_pattern, readout_train_dir, readout_test_dir) for scenario in readout_scenarios],
                }
            data_spaces.append(space)
        
    if 'all' in pretraining_protocols:
        assert len(pretraining_scenarios) > 1, 'Must have more than one scenario to do all protocol.'
        space = {
            'pretraining': build_paths('all', pretraining_scenarios, pretraining_file_pattern, pretraining_train_dir, pretraining_test_dir), # train on all scenarios
            'readout': [build_paths(scenario, [scenario], readout_file_pattern, readout_train_dir, readout_test_dir) for scenario in readout_scenarios], # readout on each scenario individually
            }
        data_spaces.append(space)

    return data_spaces

import argparse
import getpass
import glob
import socket

import os
import yaml
from hyperopt import hp
from operator import attrgetter
from physopt.config import get_cfg_defaults, get_cfg_debug

from physopt.opt import OptimizationPipeline

NO_PARAM_SPACE = hp.choice('dummy', [0])
CONFIG_ENV_VAR = 'PHYSOPT_CONFIG_DIR'
OUTPUT_ENV_VAR = 'PHYSOPT_OUTPUT_DIR'
ENV_FILE = 'environment.yaml'

def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')
    parser.add_argument('-C', '--config', required=True, type=str, help='path to physopt configuration file')
    parser.add_argument('-D', '--debug', action='store_true', help='debug mode')
    parser.add_argument('-O', '--output', type=str, help='Output directory for physopt artifacts')
    return parser.parse_args()

class MissingEnvironmentVariable(Exception):
    pass

def resolve_config_file(config_file):
    if not os.path.isabs(config_file): # skip search if abs path provided
        try:
            config_dir = os.environ[CONFIG_ENV_VAR]
        except KeyError:
            raise MissingEnvironmentVariable(f'Must set environment variable "{CONFIG_ENV_VAR}" if using relative path for config file')
        assert os.path.isdir(config_dir), f'Directory not found: {config_dir}'
        print(f'Searching for config in {config_dir}')
        pathname = os.path.join(config_dir, '**', config_file)
        files = glob.glob(pathname, recursive=True)
        assert len(files) > 0, f'No config file found matching {pathname}.'
        assert len(files) == 1, f'Found multiple ({len(files)}) files that match {pathname}'
        config_file = files[0]
    assert os.path.isfile(config_file), f'File not found: {config_file}'
    print(f'Found config file: {config_file}')
    return config_file

def resolve_output_dir(cfg_output, args_output): # updates output dir with the following priority: cmdline, environ, config (if not debug)
    if args_output is not None:
        output_dir = args_output
    elif OUTPUT_ENV_VAR in os.environ:
        output_dir = os.environ[OUTPUT_ENV_VAR]
    else:
        output_dir = cfg_output.format(getpass.getuser()) # fill in current username into path
    print(f'Output dir: {output_dir}')
    return output_dir

def check_cfg(cfg): # TODO: just check that none are none?
    attrs = [
        'DATA_SPACE.MODULE',
        'PRETRAINING.OBJECTIVE_MODULE',
        'PRETRAINING.MODEL_NAME',
        'EXTRACTION.OBJECTIVE_MODULE',
        ]
    for attr in attrs:
        retriever = attrgetter(attr)
        assert retriever(cfg) is not None, f'{attr} must be set in the config'
    if cfg.EXTRACTION.LOAD_STEP is None: # replace extraction load step with pretraining train steps, if not set
        cfg.defrost()
        cfg.EXTRACTION.LOAD_STEP = cfg.PRETRAINING.TRAIN_STEPS
        cfg.freeze()
    return True

def get_cfg_from_args(args):
    cfg = get_cfg_defaults()
    config_file  = resolve_config_file(args.config)
    cfg.merge_from_file(config_file)
    cfg.CONFIG.OUTPUT_DIR = resolve_output_dir(cfg.CONFIG.OUTPUT_DIR, args.output)
    if args.debug: # merge debug at end so takes priority
        cfg.merge_from_other_cfg(get_cfg_debug())
    cfg.freeze()
    check_cfg(cfg)
    return cfg

def setup_environment_vars():
    if os.path.isfile(ENV_FILE):
        environment = yaml.safe_load(open(ENV_FILE, 'rb'))
        hostname = socket.gethostname()
        if hostname in environment:
            assert isinstance(environment[hostname], dict)
            for k,v in environment[hostname].items():
                print(f'Setting environment variable {k} to {v}')
                os.environ[k] = str(v)

if __name__ == '__main__':
    setup_environment_vars()
    args = arg_parse()
    cfg = get_cfg_from_args(args)
    pipeline = OptimizationPipeline(cfg)
    pipeline.run()

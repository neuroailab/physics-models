from yacs.config import CfgNode as CN

_C = CN()

# data should be organized as [DIR]/[SCENARIO]/[FILE_PATTERN] (e.g. /[PRETRAINING_TRAIN_DIR]/Collide/*.hdf5)
_C.PRETRAINING_TRAIN_DIR = None
_C.PRETRAINING_TEST_DIR = None
_C.PRETRAINING_FILE_PATTERN = '*.hdf5'
_C.PRETRAINING_SCENARIOS = ['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support']
_C.PRETRAINING_PROTOCOLS = ['all', 'abo', 'only']

_C.READOUT_TRAIN_DIR = None
_C.READOUT_TEST_DIR = None
_C.READOUT_FILE_PATTERN = None
_C.READOUT_SCENARIOS = None
_C.READOUT_PROTOCOL = 'minimal' # {'full'|'minimal'}

_C.SEEDS = 1

def get_cfg_defaults():
  return _C.clone()


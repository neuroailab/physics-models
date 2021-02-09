from yacs.config import CfgNode as CN

_C = CN()

# Data Provider Params
_C.DATA = CN()
_C.DATA.MAP_PCALL_NUM = 4
_C.DATA.BATCH_SIZE = 1 
_C.DATA.ENQUEUE_BATCH_SIZE = 256
_C.DATA.BUFFER_SIZE = 16
_C.DATA.MAIN_SOURCE_KEY = 'images'
_C.DATA.SHIFTS = (30, 1024, 1)

_C.DATA.SEQ_LEN = 10
_C.DATA.IMSIZE = 224


def get_cfg_defaults():
    return _C.clone()

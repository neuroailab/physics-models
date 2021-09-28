from yacs.config import CfgNode as CN

# Config for frozen physion models

_C = CN()

_C.TRAIN_STEPS = 100
_C.BATCH_SIZE = 32

_C.TRAIN = CN()
_C.TRAIN.LR = 1e-2

_C.DATA = CN()
_C.DATA.STATE_LEN = 7 # number of images as input
_C.DATA.SEQ_LEN = 25
_C.DATA.IMSIZE = 224
_C.DATA.SUBSAMPLE_FACTOR = 6

def get_frozen_physion_cfg(debug=False):
    C =  _C.clone()
    if debug:
        C.TRAIN_STEPS = 5
    return C

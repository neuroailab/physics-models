from yacs.config import CfgNode as CN

# Config for frozen physion models

_C = CN()

# Train
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 16

_C.DATA = CN()
_C.DATA.STATE_LEN = 7 # number of images as input
_C.DATA.SEQ_LEN = 25
_C.DATA.IMSIZE = 224
_C.DATA.SUBSAMPLE_FACTOR = 6

def get_frozen_physion_cfg():
    C =  _C.clone()
    return C

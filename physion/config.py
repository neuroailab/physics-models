from yacs.config import CfgNode as CN

# Config for frozen physion models

_C = CN()

# Train
_C.EPOCHS = 20
_C.BATCH_SIZE = 16

# Model
_C.STATE_LEN = 7 # number of images as input
_C.SEQ_LEN = 25
_C.IMSIZE = 224
_C.SUBSAMPLE_FACTOR = 6

def get_frozen_physion_cfg():
    C =  _C.clone()
    return C

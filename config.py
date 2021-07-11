from yacs.config import CfgNode as CN

# Config for frozen physion models

_C = CN()

# Train
_C.EPOCHS = 20
_C.BATCH_SIZE = 16
_C.LR = 1e-3

# Model
_C.STATE_LEN = 7 # number of images as input
_C.SEQ_LEN = 25
_C.IMSIZE = 224

def get_frozen_physion_cfg(debug=False):
    C =  _C.clone()
    if debug:
        C.EPOCHS = 1
    return C

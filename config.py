from yacs.config import CfgNode as CN

# Config for frozen physion models

_C = CN()

# Train
_C.EPOCHS = 10
_C.BATCH_SIZE = 64
_C.LR = 1e-3

# Model
_C.DATA = CN() # also used for dataprovider
_C.DATA.STATE_LEN = 4 # number of images as input
_C.DATA.SEQ_LEN = 10
_C.DATA.IMSIZE = 224

def get_frozen_physion_cfg(debug=False):
    C =  _C.clone()
    if debug:
        C.EPOCHS = 1
        C.BATCH_SIZE = 16
    return C

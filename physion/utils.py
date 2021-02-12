import numpy as np
import torch

def get_subsets_from_datasets(datasets): # TODO: move to utils in physion package?
    assert isinstance(datasets, list)
    return [subset.split("/")[-1] for subset in datasets]
    

def init_seed(seed): # TODO: move to utils in physion package?
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


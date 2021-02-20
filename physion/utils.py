import numpy as np
import torch

def get_subsets_from_datasets(datasets):
    assert isinstance(datasets, list)
    return [subset.split("/")[-1] for subset in datasets]
    

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


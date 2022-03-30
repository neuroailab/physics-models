import os
import io
from os.path import expanduser
import boto3
import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from physion.objective.FitVid import save_vis
from physion.models.fitvid import FitVid
from modulefinder import ModuleFinder

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

PRETRAINING_CFG = dotdict({
    'DATA': dotdict({
        'SEQ_LEN': 16,
        'STATE_LEN': 5,
        'SUBSAMPLE_FACTOR': 9,
        'IMSIZE': 64,
        }),
    'BATCH_SIZE': 1,
    })

FITVID_CFG = {
    'input_size': 3,
    'n_past': 5,
    'z_dim': 10,
    'beta': 1.e-4,
    'g_dim': 128,
    'rnn_size': 256,
    'num_channels': 64,
    }

def get_basedir():
    home = os.path.join(expanduser("~"), ".fitvid")
    if not os.path.exists(home):
        os.makedirs(home)
    return home

def load_fitvid():
    home = get_basedir()
    modelpath = os.path.join(home, "model.pt")

    if not os.path.exists(modelpath):
        s3 = boto3.resource('s3')
        s3.Bucket('physion-physopt').download_file('02bae92d9a4d4fa98537548ca80aa53a/artifacts/step_400000/model_ckpts/model.pt', modelpath)

    model = FitVid(**FITVID_CFG)
    model.load_state_dict(torch.load(modelpath))
    return model

def test_load_fitvid(): # test pretrained fitvid on example video
    # download example video
    home = get_basedir()
    hdf5path = os.path.join(home, 'physion_example.hdf5')
    s3 = boto3.resource('s3')
    s3.Bucket('human-physics-benchmarking-towers-redyellow-pilot').download_file('pilot_towers_nb4_SJ025_mono1_dis0_occ0_tdwroom-redyellow_0011.hdf5', hdf5path)

    # preprocess input
    images = []
    with h5py.File(hdf5path, 'r') as f:
        frames = list(f['frames'])
        img_transforms = transforms.Compose([
            transforms.Resize((PRETRAINING_CFG.DATA.IMSIZE, PRETRAINING_CFG.DATA.IMSIZE)),
            transforms.ToTensor(),
            ])
        for frame in frames[:PRETRAINING_CFG.DATA.SEQ_LEN*PRETRAINING_CFG.DATA.SUBSAMPLE_FACTOR:PRETRAINING_CFG.DATA.SUBSAMPLE_FACTOR]:
            img = f['frames'][frame]['images']['_img'][()]
            img = Image.open(io.BytesIO(img)) # (256, 256, 3)
            img = img_transforms(img)
            images.append(img)
        images = torch.stack(images, dim=0)
        images = torch.unsqueeze(images, 0) # add batch dim

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = load_fitvid().to(device)
    model.eval()
    with torch.no_grad():
        model_output = model(images.to(device))

    # save visualizations
    frames = {
        'gt': images,
        'sim': model_output['preds'].cpu().detach(),
        'stimulus_name': np.array(['example_video']),
        }
    save_vis(frames, PRETRAINING_CFG, home, 0, None)

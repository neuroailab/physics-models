import os
import io
import h5py
import json
from PIL import Image
import numpy as np
import logging
import torch
from  torch.utils.data import Dataset

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

class TDWDataset(Dataset):
    def __init__(
        self,
        data_root,
        imsize,
        seq_len,
        state_len,
        train=True,
        debug=False,
        ):
        assert isinstance(data_root, list)
        self.imsize = imsize
        self.seq_len = seq_len
        self.state_len = state_len # not necessarily always used
        assert self.seq_len > self.state_len, 'Sequence length {} must be greater than state length {}'.format(self.seq_len, self.state_len)
        self.train = train
        self.debug = debug

        self.hdf5_files = []
        self.metadata = []
        for path in data_root:
            files = [os.path.join(path, f) for f in os.listdir(path)]
            self.hdf5_files.extend(sorted(list(filter(lambda f: f.endswith('.hdf5'), files)))) # sorted list of all hdf5 files
            with open(os.path.join(path, 'metadata.json')) as f:
                self.metadata.extend(json.load(f)) # assumes metadata is sorted by stimulus name
        assert len(self.metadata) == len(self.hdf5_files), 'Legnth of metadata should match hdf5 files'
        self.N = len(self.metadata)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get_seq(index)

    def get_seq(self, index):
        with h5py.File(self.hdf5_files[index], 'r') as f: # load ith hdf5 file from list
            # extract images and labels
            images = []
            labels = []
            for frame in f['frames']:
                img = f['frames'][frame]['images']['_img'][()]
                img = np.array(Image.open(io.BytesIO(img))) # (256, 256, 3)
                images.append(img)
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                labels.append(lbl)
        images = torch.from_numpy(np.array(images))
        labels = torch.from_numpy(np.array(labels))

        subsample_factor = 3 # subsample images by 3x - 30fps => 10 fps TODO: make param
        images = images[::subsample_factor]
        images = images.float().permute(0, 3, 1, 2) # (T, 3, D, D)
        images = torch.nn.functional.interpolate(images, size=self.imsize)

        labels = labels[::subsample_factor]
        labels = torch.unsqueeze(labels, -1)

        if self.train: # randomly sample sequence of seq_len
            start_idx = torch.randint(0, images.shape[0]-self.seq_len+1, (1,))[0]
            images = images[start_idx:start_idx+self.seq_len]
            labels = labels[start_idx:start_idx+self.seq_len]
        else: # get last seq_len # of frames
            images = images[-self.seq_len:]
            labels = labels[-self.seq_len:]
        # print(len(labels.numpy()), np.sum(labels.numpy()))

        sample = {
            'images': images,
            'binary_labels': labels,
        }
        # TODO: add human_prob
        return sample
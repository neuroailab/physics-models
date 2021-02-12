import os
import copy
import pickle
import torch
import tensorflow as tf
from physion.data.tfdata import SequenceNewDataProvider as DataProvider

def filter_rule(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_acting']), keys
    return tf.logical_and(data['is_moving'], tf.logical_not(data['is_acting']))

DEFAULT_DATA_PARAMS = {
    'sources': ['images', 'reference_ids', 'object_data'],
    'filter_rule': (filter_rule, ['is_moving', 'is_acting']),
    'enqueue_batch_size': 256,
    'buffer_size': 16,
    'map_pcall_num': 4,
    'shuffle': False, # shuffling will be done by pytorch dataprovider
    'use_legacy_sequence_mode': True,
    'main_source_key': 'images',
}

class TDWDataset(object):
    """Data handler which loads the TDW images."""

    def __init__(
            self,
            data_root, 
            data_cfg,
            train=True,
            ):
        self.train = train
        self.label_key = data_cfg.DATA.LABEL_KEY
        self.imsize = data_cfg.IMSIZE
        self.seq_len = data_cfg.SEQ_LEN
        self.shift_selector = slice(*data_cfg.DATA.SHIFTS)

        self._set_size(data_cfg)
        self._set_datapaths(data_root)
        self.data = self.build_data()

    def _set_datapaths(self, data_root):
        if not isinstance(data_root, list):
            data_root = [data_root]
        if self.train:
            self.datapaths = [os.path.join(dir, 'new_tfdata') for dir in data_root]
        else:
            self.datapaths = [os.path.join(dir, 'new_tfvaldata') for dir in data_root]

    def _set_size(self, data_cfg):
        if self.train:
            self.N = data_cfg.DATA.TRAIN_SIZE
        else:
            self.N = data_cfg.DATA.TEST_SIZE
        print('Dataset size: {}'.format(self.N))

    def __len__(self):
        return self.N

    def build_data(self):
        print('Building TF Dataset')
        tfdata_params = copy.deepcopy(DEFAULT_DATA_PARAMS)
        tfdata_params['sources'].append(self.label_key)
        tfdata_params['data'] = self.datapaths
        tfdata_params['sequence_len'] = self.seq_len
        tfdata_params['shift_selector'] = self.shift_selector

        data_provider = DataProvider(**tfdata_params)
        batch_size = 1 # only use bs=1 since get_seq gets once sample at a time
        dataset = data_provider.build_datasets(batch_size)
        data = iter(dataset)
        return data

    def get_seq(self, index):
        try:
            batch = self.data.get_next()
        except StopIteration:
            print('End of TF Dataset')
            # self.data = build_data(self.DATA_PARAMS)
            # batch = next(self.data)
            raise
        batch_images = batch['images'][0] # [seq_len, image_size, image_size, 3]
        batch_labels = torch.from_numpy(batch[self.label_key][0].numpy()) # (seq_len, 2) TODO key
        image_seq = torch.from_numpy(batch_images.numpy()).float().permute(0, 3, 1, 2) # (T, 3, D, D)
        image_seq = torch.nn.functional.interpolate(image_seq, size=self.imsize)
        sample = {
            'images': image_seq,
            'binary_labels': batch_labels,
            }

        return sample

    def __getitem__(self, index):
        return self.get_seq(index)

class TDWHumanDataset(object): # TODO: use common base class with TDWDatasetj?
    def __init__(
            self,
            data_root, 
            data_cfg,
            ):
        self.label_key = data_cfg.DATA.LABEL_KEY
        self.imsize = data_cfg.IMSIZE

        self._set_datapaths(data_root)
        data = self.build_data()
        self.N = len(data) # must do before converting to iterator
        print('Dataset size: {}'.format(self.N))
        self.data = iter(data) # could probably also use list and index into it in __getitem__

    def _set_datapaths(self, data_root):
        if not isinstance(data_root, list):
            data_root = [data_root]
        self.datapaths = [os.path.join(path, 'raw_data.pickle') for path in data_root]

    def build_data(self):
        data = []
        for path in self.datapaths :
            data.extend(pickle.load(open(path, 'rb')))
        return data

    def __len__(self):
        return self.N

    def get_seq(self, index):
        data = next(self.data)
        images = data['images'][0] # (10, 256, 256, 3)
        binary_labels = torch.from_numpy(data[self.label_key][0]) # (10, ...)
        # TODO: add human_prob
        image_seq = torch.from_numpy(images).float().permute(0, 3, 1, 2) # (T, 3, D, D)
        image_seq = torch.nn.functional.interpolate(image_seq, size=self.imsize) # (T, 3, D', D')
    
        sample = {
            'images': image_seq,
            'binary_labels': binary_labels,
        }
        return sample

    def __getitem__(self, index):
        return self.get_seq(index)

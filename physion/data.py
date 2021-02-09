import os
import copy
import pickle
import torch
import tensorflow as tf
from physion.tfdata_provider.data_provider import SequenceNewDataProvider as DataProvider

def filter_rule(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_acting']), keys
    return tf.logical_and(data['is_moving'], tf.logical_not(data['is_acting']))

DEFAULT_SOURCES = ['images', 'reference_ids', 'object_data']
DEFAULT_FILTER_RULE = (filter_rule, ['is_moving', 'is_acting'])

class TDWDataset(object):
    """Data handler which loads the TDW images."""

    def __init__(
            self,
            data_root, 
            label_key, 
            data_cfg,
            size=None, # TODO: put into cfg?
            train=True,
            ):
        self.cfg = data_cfg
        self.label_key = label_key
        self.sources = DEFAULT_SOURCES + [label_key]
        if not isinstance(data_root, list):
            data_root = [data_root]
        if train:
            self.datapaths = [os.path.join(dir, 'new_tfdata') for dir in data_root]
        else:
            self.datapaths = [os.path.join(dir, 'new_tfvaldata') for dir in data_root]
        self.data = self.build_data()
        if size is not None:
            self.N = size
        else:
            self.N = 10000 # 57856 # TODO
        self.imsize = data_cfg.IMSIZE

    def __len__(self):
        return self.N

    def build_data(self):
        print('Building TF Dataset')
        data_provider = DataProvider(
            data=self.datapaths,
            enqueue_batch_size=self.cfg.ENQUEUE_BATCH_SIZE,
            sources=self.sources,
            sequence_len=self.cfg.SEQ_LEN,
            shift_selector=slice(*self.cfg.SHIFTS),
            buffer_size=self.cfg.BUFFER_SIZE,
            filter_rule=DEFAULT_FILTER_RULE,
            seed=0, # shouldn't actually be necessary since we're not shuffling
            shuffle=False, # shuffling will be done by pytorch dataprovider
            use_legacy_sequence_mode=True,
        )
        assert self.cfg.BATCH_SIZE == 1
        dataset = data_provider.build_datasets(self.cfg.BATCH_SIZE)
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

class TDWHumanDataset(object):
    def __init__(
            self,
            data_root, 
            label_key, 
            data_cfg, # only used for imsize
            ):
        self.label_key = label_key
        if not isinstance(data_root, list):
            data_root = [data_root]
        self.datapaths = [os.path.join(path, 'raw_data.pickle') for path in data_root]
        data = []
        for path in self.datapaths :
            data.extend(self.build_data(path))
        self.N = len(data) # must do before converting to iterator
        self.data = iter(data) # could probably also use list and index into it in __getitem__
        self.imsize = data_cfg.IMSIZE

    @staticmethod
    def build_data(path):
        dataset = pickle.load(open(path, 'rb'))
        return dataset

    def __len__(self): # TODO: use common base class?
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

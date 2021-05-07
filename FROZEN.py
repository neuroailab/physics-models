import argparse
import datetime
import os
import pickle
import numpy as np
import logging
from hyperopt import STATUS_OK

from physopt.utils import PhysOptObjective
import physion.modules.frozen as modules
from physopt.models.physion.config import get_frozen_physion_cfg
from physion.data.pydata import TDWDataset, TDWHumanDataset
from physion.data.new_pydata import TDWDataset as NewTDWDataset
from physion.data.config import get_data_cfg
from physion.utils import init_seed, get_subsets_from_datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import torch.optim as optim
import tensorflow as tf

def get_model(encoder, dynamics):
    return modules.FrozenPhysion(encoder, dynamics)

def get_dataset(dataset, human=False):
    if dataset == 'new':
        return NewTDWDataset
    elif human:
        return TDWHumanDataset
    else:
        return TDWDataset

def train(config):
    device = config['device']
    model =  config['model']
    state_len = config['state_len']

    if os.path.isfile(config['model_file']): # load existing model ckpt
        model.load_state_dict(torch.load(config['model_file']))
        logging.info('Loaded existing ckpt')
    else:
        logging.info('Training from scratch')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    Dataset = get_dataset(config['dataset'])
    dataset = Dataset(
        data_root=config['datapaths'],
        imsize=config['imsize'],
        seq_len=config['seq_len'],
        state_len=config['state_len'],
        debug=config['debug'],
        )
    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # save initial model
    torch.save(model.state_dict(), config['model_file'])
    print('Saved model checkpoint to: {}'.format(config['model_file']))

    best_loss = 1e9
    for epoch in range(config['epochs']):
        print('Staring epoch {}/{}'.format(epoch+1, config['epochs']))
        running_loss = 0.
        for i, data in enumerate(trainloader):
            images = data['images'].to(device)
            inputs = images[:,:state_len] # TODO: have state_len accounted for in dataprovider
            labels = model.get_encoder_feats(images[:,state_len])
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss/(i+1)
            print(avg_loss)

        # save model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['model_file'])
            logging.info('Saving model with loss {} at epoch {}'.format(best_loss, epoch))
            print('Saved model checkpoint to: {}'.format(config['model_file']))

def test(config):
    device = config['device']
    model =  config['model']
    state_len = config['state_len']

    model.load_state_dict(torch.load(config['model_file'])) # load weights
    model.eval() # set to eval mode

    Dataset = get_dataset(config['dataset'], 'human' in config['name'])
    dataset = Dataset(
        data_root=config['datapaths'],
        imsize=config['imsize'],
        seq_len=config['seq_len'],
        state_len=config['state_len'],
        debug=config['debug'],
        train=False, # TODO
        )
    testloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    extracted_feats = []

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images = data['images'].to(device)

            encoded_states = model.get_encoder_feats(images)
            rollout_states = encoded_states[:state_len] # copy over feats for seed frames
            rollout_steps = images.shape[1] - state_len 

            for step in range(rollout_steps):
                input_feats = rollout_states[-state_len:]
                pred_state  = model.dynamics(input_feats) # dynamics model predicts next latent from past latents
                rollout_states.append(pred_state)

            encoded_states = torch.stack(encoded_states, axis=1).cpu().numpy() # TODO: cpu vs detach?
            rollout_states = torch.stack(rollout_states, axis=1).cpu().numpy()
            labels = data['binary_labels'].cpu().numpy()
            print(encoded_states.shape, rollout_states.shape, labels.shape)
            extracted_feats.append({
                'encoded_states': encoded_states,
                'rollout_states': rollout_states,
                'binary_labels': labels,
            })

    pickle.dump(extracted_feats, open(config['feature_file'], 'wb')) 
    print('Saved features to {}'.format(config['feature_file']))

class Objective(PhysOptObjective):
    def __init__(self,
            exp_key,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat,
            debug,
            encoder,
            dynamics,
            ):
        super().__init__(exp_key, seed, train_data, feat_data, output_dir, extract_feat, debug)
        self.encoder = encoder
        self.dynamics = dynamics

    def __call__(self, *args, **kwargs):
        results = super().__call__()
        cfg = get_frozen_physion_cfg(debug=self.debug)
        cfg.freeze()

        model_file = os.path.join(self.model_dir, 'model.pt') # TODO: move to PhysOptObjective?
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model =  get_model(self.encoder, self.dynamics).to(device)
        # model = nn.DataParallel(model) # TODO: multi-gpu doesn't work
        config = {
            'name': self.feat_data['name'],
            'datapaths': self.feat_data['data'],
            'dataset': 'new', # TODO
            'device': device, 
            'model': model,
            'epochs': cfg.EPOCHS,
            'batch_size': cfg.BATCH_SIZE,
            'lr': cfg.LR,
            'model_file': model_file,
            'feature_file': self.feature_file,
            'imsize': cfg.IMSIZE,
            'seq_len': cfg.SEQ_LEN,
            'state_len': cfg.STATE_LEN, # number of images as input
            'debug': self.debug, # for dp
        }
        init_seed(self.seed)

        if self.extract_feat: # save out model features from trained model
            test(config) 
        else: # run model training
            train(config)

        results['loss'] = 0.
        results['encoder'] = self.encoder
        results['dynamics'] = self.dynamics
        return results

class VGGFrozenIDObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', dynamics='id')

class VGGFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', dynamics='mlp')

class VGGFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', dynamics='lstm')

class DEITFrozenIDObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', dynamics='id')

class DEITFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', dynamics='mlp')

class DEITFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', dynamics='lstm')

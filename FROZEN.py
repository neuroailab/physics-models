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
from physion.data.config import get_data_cfg
from physion.utils import init_seed, get_subsets_from_datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import torch.optim as optim
import tensorflow as tf

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    encoder='vgg',
    dynamics='lstm',
    feature_file=None,
    debug=False,
    ):
    cfg = get_frozen_physion_cfg(debug=debug)
    cfg.freeze()

    model_file = os.path.join(model_dir, 'model.pt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model =  get_model(encoder, dynamics).to(device)
    # model = nn.DataParallel(model) # TODO: multi-gpu doesn't work
    config = {
        'name': name,
        'datapaths': datasets,
        'device': device, 
        'encoder': encoder,
        'dynamics': dynamics,
        'model': model,
        'epochs': cfg.EPOCHS,
        'batch_size': cfg.BATCH_SIZE,
        'lr': cfg.LR,
        'model_file': model_file,
        'feature_file': feature_file,
        'imsize': cfg.IMSIZE,
        'seq_len': cfg.SEQ_LEN,
        'state_len': cfg.STATE_LEN, # number of images as input
        'debug': debug,
    }
    init_seed(seed)
    if write_feat:
        test(config)
    else:
        train(config)

def get_model(encoder, dynamics):
    return modules.FrozenPhysion(encoder, dynamics)

def train(config):
    device = config['device']
    model =  config['model']
    state_len = config['state_len']

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    dataset = TDWDataset(
        data_root=config['datapaths'],
        imsize=config['imsize'],
        seq_len=config['seq_len'],
        state_len=config['state_len'],
        debug=config['debug'],
        )
    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    best_loss = 1e9
    for epoch in range(config['epochs']):
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
            print('Saved model checkpoint to: {}'.format(config['model_file']))

def test(config):
    device = config['device']
    model =  config['model']
    state_len = config['state_len']

    # load weights
    model.load_state_dict(torch.load(config['model_file']))
    model.eval()

    if 'human' in config['name']:
        dataset = TDWHumanDataset(
            data_root=config['datapaths'],
            imsize=config['imsize'],
            seq_len=config['seq_len'],
            state_len=config['state_len'],
            )
    else:
        dataset = TDWDataset(
            data_root=config['datapaths'],
            imsize=config['imsize'],
            seq_len=config['seq_len'],
            state_len=config['state_len'],
            debug=config['debug'],
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
        if self.extract_feat: # save out model features from trained model
            write_feat = 'human' if 'human' in self.feat_data['name'] else 'train'
            run(
                name=self.feat_data['name'],
                datasets=self.feat_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                write_feat=write_feat,
                encoder=self.encoder,
                dynamics=self.dynamics,
                feature_file=self.feature_file,
                debug=self.debug,
                ) 

        else: # run model training
            run(
                name=self.train_data['name'],
                datasets=self.train_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                encoder=self.encoder,
                dynamics=self.dynamics,
                debug=self.debug,
                )

        return {
                'loss': 0.0,
                'status': STATUS_OK,
                'exp_key': self.exp_key,
                'seed': self.seed,
                'train_data': self.train_data,
                'feat_data': self.feat_data,
                'model_dir': self.model_dir,
                }

class VGGFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', dynamics='mlp')

class VGGFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', dynamics='lstm')

class DEITFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', dynamics='mlp')

class DEITFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', dynamics='lstm')

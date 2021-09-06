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
from physopt.models.pydata import TDWDataset
from physopt.models.config import get_data_cfg 
from physion.utils import init_seed, get_subsets_from_datasets
from torch.utils.data import DataLoader
import torch
import mlflow

class Objective(PhysOptObjective):
    def __init__(self,
            exp_key,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat,
            debug,
            max_run_time,
            encoder,
            dynamics,
            ):
        super().__init__(exp_key, seed, train_data, feat_data, output_dir, extract_feat, debug)
        self.encoder = encoder
        self.dynamics = dynamics
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = '{}Frozen{}'.format(encoder, dynamics) # mlflow experiment name tied to the Objective class TODO: or maybe just have everything in one experiment, and only use run_name

    def __call__(self, *args, **kwargs):
        results = super().__call__()
        cfg = get_frozen_physion_cfg(debug=self.debug)
        cfg.freeze()

        config = {
            'epochs': cfg.EPOCHS,
            'batch_size': cfg.BATCH_SIZE,
            'lr': cfg.LR,
            'feature_file': self.feature_file,
            'imsize': cfg.IMSIZE,
            'seq_len': cfg.SEQ_LEN,
            'state_len': cfg.STATE_LEN, # number of images as input
            'debug': self.debug, # for dp
        }
        logging.info(config)
        init_seed(self.seed) # TODO

        if self.extract_feat: # save out model features from trained model
            config['datapaths'] = self.feat_data['data'] # TODO
            self.test(config) 
        else: # run model training
            config['datapaths'] = self.train_data['data'] # TODO
            self.train(config)

        results['loss'] = 0.
        results['encoder'] = self.encoder
        results['dynamics'] = self.dynamics
        return results

    def train(self, config):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.exp_key)

        model = self.get_model()
        model = self.load_model(model)

        trainloader = self.get_dataloader(config, train=True)
        best_loss = 1e9
        for epoch in range(config['epochs']):
            logging.info('Starting epoch {}/{}'.format(epoch+1, config['epochs']))
            running_loss = 0.
            for i, data in enumerate(trainloader):
                loss = self.train_step(data, model, config)
                running_loss += loss
                avg_loss = running_loss/(i+1)
                print(avg_loss)
            mlflow.log_metric(key='avg_loss', value=avg_loss, step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            logging.info('Saving model with loss {} at epoch {}'.format(best_loss, epoch))
            self.save_model(model)
        mlflow.end_run()

    def test(self, config):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.exp_key)

        model = self.get_model()
        assert os.path.isfile(self.model_file), 'No model ckpt found, cannot extract features'
        model = self.load_model(model)
        model.eval() # set to eval mode

        testloader = self.get_dataloader(config, train=False)
        extracted_feats = []
        for i, data in enumerate(testloader):
            output = self.test_step(data, model, config)
            extracted_feats.append(output)

        pickle.dump(extracted_feats, open(config['feature_file'], 'wb')) 
        print('Saved features to {}'.format(config['feature_file']))
        mlflow.end_run()

    def get_dataloader(self, config, train=True):
        dataset = TDWDataset(
            data_root=config['datapaths'],
            imsize=config['imsize'],
            seq_len=config['seq_len'],
            state_len=config['state_len'],
            debug=config['debug'],
            train=train,
            )
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=train)
        return dataloader

    def get_model(self):
        model =  modules.FrozenPhysion(self.encoder, self.dynamics).to(self.device)
        # model = torch.nn.DataParallel(model) # TODO: multi-gpu doesn't work yet, also for loading
        return model

    def load_model(self, model):
        if os.path.isfile(self.model_file): # load existing model ckpt TODO: add option to disable reloading
            model.load_state_dict(torch.load(self.model_file))
            logging.info('Loaded existing ckpt')
        else:
            torch.save(model.state_dict(), self.model_file) # save initial model
            logging.info('Training from scratch')
        return model

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_file)
        logging.info('Saved model checkpoint to: {}'.format(self.model_file))

    def train_step(self, data, model, config):
        images = data['images'].to(self.device)
        inputs = images[:,:config['state_len']] # TODO: have state_len accounted for in dataprovider
        labels = model.get_encoder_feats(images[:,config['state_len']])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    def test_step(self, data, model, config):
        with torch.no_grad():
            images = data['images'].to(self.device)

            encoded_states = model.get_encoder_feats(images)
            rollout_states = encoded_states[:config['state_len']] # copy over feats for seed frames
            rollout_steps = images.shape[1] - config['state_len'] 

            for step in range(rollout_steps):
                input_feats = rollout_states[-config['state_len']:]
                pred_state  = model.dynamics(input_feats) # dynamics model predicts next latent from past latents
                rollout_states.append(pred_state)

        encoded_states = torch.stack(encoded_states, axis=1).cpu().numpy() # TODO: cpu vs detach?
        rollout_states = torch.stack(rollout_states, axis=1).cpu().numpy()
        labels = data['binary_labels'].cpu().numpy()
        stimulus_name = data['stimulus_name']
        print(encoded_states.shape, rollout_states.shape, labels.shape)
        output = {
            'encoded_states': encoded_states,
            'rollout_states': rollout_states,
            'binary_labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output

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

class CLIPFrozenIDObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='clip', dynamics='id')

class CLIPFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='clip', dynamics='mlp')

class CLIPFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='clip', dynamics='lstm')

class DINOFrozenIDObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='dino', dynamics='id')

class DINOFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='dino', dynamics='mlp')

class DINOFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='dino', dynamics='lstm')

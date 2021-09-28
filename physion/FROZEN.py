import numpy as np
import logging
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from physion.utils import PytorchPhysOptObjective
from physion.data.pydata import TDWDataset as TDWDatasetBase

import physion.models.frozen as models
from physion.config import get_frozen_physion_cfg

class TDWDataset(TDWDatasetBase): # TODO: move to physion.utils
    def __getitem__(self, index):
        sample = self.get_seq(index)
        images = sample['images'] # (seq_len, 3, D', D')
        input_images = images[:self.state_len]
        label_image = images[self.state_len]
        sample.update({
            'input_images': input_images,
            'label_image': label_image,
            })
        return sample

class Objective(PytorchPhysOptObjective):
    def get_config(self):
        cfg = super().get_config()
        cfg.defrost()
        cfg.merge_from_other_cfg(get_frozen_physion_cfg())
        cfg.freeze()
        return cfg

    def get_dataloader(self, datapaths, phase, shuffle, **kwargs):
        cfg = self.cfg
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.DATA.IMSIZE,
            seq_len=cfg.DATA.SEQ_LEN,
            state_len=cfg.DATA.STATE_LEN,
            random_seq=True if phase=='pretraining' else False,
            debug=self.debug,
            subsample_factor=cfg.DATA.SUBSAMPLE_FACTOR
            )
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle)
        return dataloader

    def train_step(self, data):
        self.model.train() # set to train mode
        inputs = data['input_images'].to(self.device)
        labels = self.model.get_encoder_feats(data['label_image'].to(self.device))
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9) # TODO: add these to cfg
        optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    def validation(self):
        valloader = self.get_dataloader(self.pretraining_space['test'], phase='pretraining', train=False, shuffle=False)
        val_results = []
        pred_states = []
        next_states = []
        with torch.no_grad():
            for i, data in enumerate(valloader):
                logging.info('Val Step: {0:>5}'.format(i+1))
                val_res = self.val_step(data)
                assert isinstance(val_res, dict)
                val_results.append(val_res)

                state_len = data['input_images'].shape[1]
                states = self.model.get_encoder_feats(data['images'][:,:state_len+1].to(self.device))
                input_state = states[:-1]
                next_state = states[-1]
                pred_state  = self.model.dynamics(input_state) # dynamics model predicts next latent from past latents

                pred_states.append(pred_state.cpu())
                next_states.append(next_state.cpu())

            pred_state_cat = torch.cat(pred_states, dim=0)
            next_state_cat = torch.cat(next_states, dim=0)

        # convert list of dicts into single dict by aggregating with mean over values for a given key
        val_results = {k: np.mean([res[k] for res in val_results]) for k in val_results[0]} # assumes all keys are the same across list
        val_results.update(self.latent_eval(pred_state_cat, next_state_cat))
        return val_results

    def latent_eval(self, pred_state_cat, next_state_cat): # TODO
        topk = [1]
        hits_at = defaultdict(int)
        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        dist_matrix = pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples = full_size
        print('Size of current topk evaluation batch: {}'.format(
            full_size))

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum = reciprocal_ranks.sum().item()

        for k in topk:
            print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

        print('MRR: {}'.format(rr_sum / float(num_samples)))

        val_results = {
            'Hits_at_1': hits_at[1] / float(num_samples),
            'MRR': rr_sum / float(num_samples),
            'num_samples': num_samples,
            }
        return val_results

    def val_step(self, data): # TODO: reduce duplication with train_step
        self.model.eval() # set to eval mode
        inputs = data['input_images'].to(self.device)
        labels = self.model.get_encoder_feats(data['label_image'].to(self.device))
        outputs = self.model(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, labels)
        return {'val_loss': loss.item()}

    def extract_feat_step(self, data):
        self.model.eval() # set to eval mode
        with torch.no_grad(): # TODO: could use a little cleanup
            state_len = data['input_images'].shape[1]
            input_states = self.model.get_encoder_feats(data['input_images'].to(self.device))
            images = data['images'][:, state_len:].to(self.device)
            observed_states = self.model.get_encoder_feats(images)
            rollout_steps = images.shape[1]

            simulated_states = []
            prev_states = input_states
            for step in range(rollout_steps):
                pred_state  = self.model.dynamics(prev_states) # dynamics model predicts next latent from past latents
                simulated_states.append(pred_state)
                # add most recent pred and delete oldest
                prev_states.append(pred_state)
                prev_states.pop(0)

        input_states = torch.stack(input_states, axis=1).cpu().numpy()
        observed_states = torch.stack(observed_states, axis=1).cpu().numpy() # TODO: cpu vs detach?
        simulated_states = torch.stack(simulated_states, axis=1).cpu().numpy()
        labels = data['binary_labels'].cpu().numpy()
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        output = {
            'input_states': input_states,
            'observed_states': observed_states,
            'simulated_states': simulated_states,
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output

def get_frozen_model(encoder, dynamics):
    model =  models.FrozenPhysion(encoder, dynamics)
    # model = torch.nn.DataParallel(model) # TODO: multi-gpu doesn't work yet, also for loading
    return model

class pVGG_IDObjective(Objective):
    model_name = 'pVGG_ID'
    def get_model(self):
        return get_frozen_model('vgg', 'id').to(self.device)

class pVGG_MLPObjective(Objective):
    model_name = 'pVGG_MLP'
    def get_model(self):
        return get_frozen_model('vgg', 'mlp').to(self.device)

class pVGG_LSTMObjective(Objective):
    model_name = 'pVGG_LSTM'
    def get_model(self):
        return get_frozen_model('vgg', 'lstm').to(self.device)

class pDEIT_IDObjective(Objective):
    model_name = 'pDEIT_ID'
    def get_model(self):
        return get_frozen_model('deit', 'id').to(self.device)

class pDEIT_MLPObjective(Objective):
    model_name = 'pDEIT_MLP'
    def get_model(self):
        return get_frozen_model('deit', 'mlp').to(self.device)

class pDEIT_LSTMObjective(Objective):
    model_name = 'pDEIT_LSTM'
    def get_model(self):
        return get_frozen_model('deit', 'lstm').to(self.device)

class pCLIP_IDObjective(Objective):
    model_name = 'pCLIP_ID'
    def get_model(self):
        return get_frozen_model('clip', 'id').to(self.device)

class pCLIP_MLPObjective(Objective):
    model_name = 'pCLIP_MLP'
    def get_model(self):
        return get_frozen_model('clip', 'mlp').to(self.device)

class pCLIP_LSTMObjective(Objective):
    model_name = 'pCLIP_LSTM'
    def get_model(self):
        return get_frozen_model('clip', 'lstm').to(self.device)

class pDINO_IDObjective(Objective):
    model_name = 'pDINO_ID'
    def get_model(self):
        return get_frozen_model('dino', 'id').to(self.device)

class pDINO_MLPObjective(Objective):
    model_name = 'pDINO_MLP'
    def get_model(self):
        return get_frozen_model('dino', 'mlp').to(self.device)

class pDINO_LSTMObjective(Objective):
    model_name = 'pDINO_LSTM'
    def get_model(self):
        return get_frozen_model('dino', 'lstm').to(self.device)

def pairwise_distance_matrix(x, y): # TODO
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)

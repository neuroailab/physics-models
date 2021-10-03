import numpy as np
import logging
import torch
from torch.utils.data import DataLoader

from physopt.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME
from physion.utils import PytorchPhysOptObjective
from physion.data.pydata import TDWDataset
from physion.metrics import latent_eval
import physion.models.frozen as models

class Objective(PytorchPhysOptObjective):
    def get_dataloader(self, datapaths, phase, shuffle, **kwargs):
        cfg = self.cfg
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.DATA.IMSIZE,
            seq_len=cfg.DATA.SEQ_LEN,
            state_len=cfg.DATA.STATE_LEN,
            random_seq=True if phase==PRETRAINING_PHASE_NAME else False,
            debug=self.cfg.DEBUG,
            subsample_factor=cfg.DATA.SUBSAMPLE_FACTOR,
            seed=self.seed,
            )
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=2)
        return dataloader

    def train_step(self, data):
        self.model.train() # set to train mode
        inputs = data['input_images'].to(self.device)
        labels = self.model.get_encoder_feats(data['label_image'].to(self.device))
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR)
        optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    def validation(self):
        valloader = self.get_dataloader(self.pretraining_space['test'], phase=PRETRAINING_PHASE_NAME, train=False, shuffle=False)
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

            pred_state_cat = torch.cat(pred_states, dim=0).numpy()
            next_state_cat = torch.cat(next_states, dim=0).numpy()

        # convert list of dicts into single dict by aggregating with mean over values for a given key
        val_results = {k: np.mean([res[k] for res in val_results]) for k in val_results[0]} # assumes all keys are the same across list
        val_results.update(latent_eval(pred_state_cat, next_state_cat))
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

import numpy as np
import logging
from collections import defaultdict
import torch
import torch.nn.functional as F

from physopt.objective.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME
from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.metrics import latent_eval

from cswm.modules import ContrastiveSWM
from cswm.utils import TDWDataset, weights_init

class CSWMModel(PytorchModel):
    def get_model(self):
        cfg = self.pretraining_cfg
        model = ContrastiveSWM(
            embedding_dim=cfg.MODEL.EMBEDDING_DIM,
            hidden_dim=cfg.MODEL.HIDDEN_DIM,
            action_dim=cfg.MODEL.ACTION_DIM,
            input_dims=(cfg.DATA.STATE_LEN*3, cfg.DATA.IMSIZE, cfg.DATA.IMSIZE),
            num_objects=cfg.MODEL.NUM_OBJECTS,
            sigma=cfg.MODEL.SIGMA,
            hinge=cfg.MODEL.HINGE,
            ignore_action=cfg.MODEL.IGNORE_ACTION,
            encoder=cfg.MODEL.ENCODER,
            ).to(self.device)
        model.apply(weights_init) 
        return model

class ExtractionObjective(ExtractionObjectiveBase, CSWMModel):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle)

    def extract_feat_step(self, data):
        self.model.eval() # set to eval mode

        with torch.no_grad():
            observations = [t.to(self.device) for t in data['all_obs']]
            actions = [t.to(self.device) for t in data['action']]
            for i, curr_obs in enumerate(observations):
                if i == 0:
                    state = self.model.obj_encoder(self.model.obj_extractor(curr_obs)) # (BS, no, embedding_dim)
                    pred_state = state # initialize predicted state to input state
                    input_states = [torch.flatten(state, 1)]
                    observed_states = []
                    simulated_states = []
                else:
                    obs_state = self.model.obj_encoder(self.model.obj_extractor(curr_obs))
                    observed_states.append(torch.flatten(obs_state, 1))
                    pred_trans = self.model.transition_model(pred_state, actions[0]) # just use first action since it's always 0
                    pred_state = pred_state + pred_trans
                    simulated_states.append(torch.flatten(pred_state, 1))

        # TODO: this is duplicated code across models
        input_states = torch.stack(input_states, axis=1).cpu().numpy()
        observed_states = torch.stack(observed_states, axis=1).cpu().numpy()
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

class PretrainingObjective(PretrainingObjectiveBase, CSWMModel):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle)

    def train_step(self, data):
        self.model.train() # set to train mode

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.pretraining_cfg.TRAIN.LR)
        optimizer.zero_grad()

        loss = self.get_contrastive_loss(data)
        loss.backward()
        optimizer.step()
        return loss.item() # scalar loss value for the step

    def validation(self):
        valloader = self.get_pretraining_dataloader(self.pretraining_space['test'], train=False)
        val_results = []
        pred_states = []
        next_states = []
        with torch.no_grad():
            for i, data in enumerate(valloader):
                logging.info('Val Step: {0:>5}'.format(i+1))
                val_res = self.val_step(data)
                assert isinstance(val_res, dict)
                val_results.append(val_res)

                data = [data[k] for k in ['obs', 'action', 'next_obs']] # to match format of StateTransitionsDataset
                data = [tensor.to(self.device) for tensor in data]
                obs, actions, next_obs = data
                state = self.model.obj_encoder(self.model.obj_extractor(obs))
                next_state = self.model.obj_encoder(self.model.obj_extractor(next_obs))

                pred_trans = self.model.transition_model(state, actions[0]) # just use first action since it's always 0
                pred_state = state + pred_trans

                pred_states.append(pred_state.cpu())
                next_states.append(next_state.cpu())

            pred_state_cat = torch.cat(pred_states, dim=0).numpy()
            next_state_cat = torch.cat(next_states, dim=0).numpy()

        # convert list of dicts into single dict by aggregating with mean over values for a given key
        val_results = {k: np.mean([res[k] for res in val_results]) for k in val_results[0]} # assumes all keys are the same across list
        val_results.update(latent_eval(pred_state_cat, next_state_cat))
        return val_results

    def val_step(self, data):
        self.model.eval() # set to eval mode
        loss = self.get_contrastive_loss(data)
        return {'val_loss': loss.item()}

    def get_contrastive_loss(self, data):
        data = [data[k] for k in ['obs', 'action', 'next_obs']] # to match format of StateTransitionsDataset
        data = [tensor.to(self.device) for tensor in data]
        loss = self.model.contrastive_loss(*data)
        return loss

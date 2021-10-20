import numpy as np
import logging
import torch

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective import PytorchModel
from physion.data.pydata import TDWDataset
from physion.metrics import latent_eval
import physion.models.frozen as models

def get_frozen_model(encoder, dynamics):
    model =  models.FrozenPhysion(encoder, dynamics)
    # model = torch.nn.DataParallel(model) # TODO: multi-gpu doesn't work yet, also for loading
    return model

class FrozenModel(PytorchModel):
    def get_model(self):
        model_name = self.pretraining_cfg.MODEL_NAME
        assert isinstance(model_name, str)
        assert model_name.count('_') == 1, f'model name should be of the form "p{{ENCODER}}_{{DYNAMICS}}", but is "{model_name}"'
        assert model_name[0] == 'p', f'model name should be of the form "p{{ENCODER}}_{{DYNAMICS}}", but is "{model_name}"'
        encoder, dynamics = model_name[1:].split('_')
        logging.info(f'Getting model... Encoder: {encoder.lower()} | Dynamics: {dynamics.lower()}')
        model = get_frozen_model(encoder.lower(), dynamics.lower()).to(self.device)
        return model

class ExtractionObjective(ExtractionObjectiveBase, FrozenModel):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle)

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

class PretrainingObjective(PretrainingObjectiveBase, FrozenModel):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle)

    def train_step(self, data):
        self.model.train() # set to train mode
        inputs = data['input_images'].to(self.device)
        labels = self.model.get_encoder_feats(data['label_image'].to(self.device))
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR)
        optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

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

import numpy as np
import torch
from torch.utils.data import DataLoader

from physopt.utils import PytorchPhysOptObjective
from physopt.models.pydata import TDWDataset as TDWDatasetBase

import physion.modules.frozen as modules
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
        if self.debug:
            cfg.merge_from_list(['EPOCHS', 1])
        cfg.freeze()
        return cfg

    def get_dataloader(self, datapaths, phase, shuffle, **kwargs):
        cfg = self.cfg
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.DATA.IMSIZE,
            seq_len=cfg.DATA.SEQ_LEN,
            state_len=cfg.DATA.STATE_LEN,
            random_seq=True if phase=='dynamics' else False,
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

    def val_step(self, data): # TODO: reduce duplication with train_step
        self.model.eval() # set to eval mode
        inputs = data['input_images'].to(self.device)
        labels = self.model.get_encoder_feats(data['label_image'].to(self.device))
        outputs = self.model(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, labels)
        return loss.item()

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
        for k,v in output.items():
            print(k, v.shape)
        return output

def get_frozen_model(encoder, dynamics):
    model =  modules.FrozenPhysion(encoder, dynamics)
    # model = torch.nn.DataParallel(model) # TODO: multi-gpu doesn't work yet, also for loading
    return model

class pVGG_IDObjective(Objective):
    def get_model(self):
        return get_frozen_model('vgg', 'id').to(self.device)

class pVGG_MLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('vgg', 'mlp').to(self.device)

class pVGG_LSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('vgg', 'lstm').to(self.device)

class pDEIT_IDObjective(Objective):
    def get_model(self):
        return get_frozen_model('deit', 'id').to(self.device)

class pDEIT_MLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('deit', 'mlp').to(self.device)

class pDEIT_LSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('deit', 'lstm').to(self.device)

class pCLIP_IDObjective(Objective):
    def get_model(self):
        return get_frozen_model('clip', 'id').to(self.device)

class pCLIP_MLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('clip', 'mlp').to(self.device)

class pCLIP_LSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('clip', 'lstm').to(self.device)

class pDINO_IDObjective(Objective):
    def get_model(self):
        return get_frozen_model('dino', 'id').to(self.device)

class pDINO_MLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('dino', 'mlp').to(self.device)

class pDINO_LSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('dino', 'lstm').to(self.device)

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
        cfg = get_frozen_physion_cfg()
        if self.debug:
            cfg.EPOCHS = 1
        cfg.freeze()
        return cfg

    def get_dataloader(self, datapaths, train=True):
        cfg = self.cfg
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.IMSIZE,
            seq_len=cfg.SEQ_LEN,
            state_len=cfg.STATE_LEN,
            train=train,
            debug=self.debug,
            subsample_factor=cfg.SUBSAMPLE_FACTOR
            )
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=train)
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

    def test_step(self, data):
        self.model.eval() # set to eval mode
        with torch.no_grad():
            images = data['images'].to(self.device)
            state_len = data['input_images'].shape[1] # TODO: hacky
            encoded_states = self.model.get_encoder_feats(images)
            rollout_states = encoded_states[:state_len] # copy over feats for seed frames
            rollout_steps = images.shape[1] - state_len 

            for step in range(rollout_steps):
                input_feats = rollout_states[-state_len:]
                pred_state  = self.model.dynamics(input_feats) # dynamics model predicts next latent from past latents
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

def get_frozen_model(encoder, dynamics):
    model =  modules.FrozenPhysion(encoder, dynamics)
    # model = torch.nn.DataParallel(model) # TODO: multi-gpu doesn't work yet, also for loading
    return model

class VGGFrozenIDObjective(Objective):
    def get_model(self):
        return get_frozen_model('vgg', 'id').to(self.device)

class VGGFrozenMLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('vgg', 'mlp').to(self.device)

class VGGFrozenLSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('vgg', 'lstm').to(self.device)

class DEITFrozenIDObjective(Objective):
    def get_model(self):
        return get_frozen_model('deit', 'id').to(self.device)

class DEITFrozenMLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('deit', 'mlp').to(self.device)

class DEITFrozenLSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('deit', 'lstm').to(self.device)

class CLIPFrozenIDObjective(Objective):
    def get_model(self):
        return get_frozen_model('clip', 'id').to(self.device)

class CLIPFrozenMLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('clip', 'mlp').to(self.device)

class CLIPFrozenLSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('clip', 'lstm').to(self.device)

class DINOFrozenIDObjective(Objective):
    def get_model(self):
        return get_frozen_model('dino', 'id').to(self.device)

class DINOFrozenMLPObjective(Objective):
    def get_model(self):
        return get_frozen_model('dino', 'mlp').to(self.device)

class DINOFrozenLSTMObjective(Objective):
    def get_model(self):
        return get_frozen_model('dino', 'lstm').to(self.device)

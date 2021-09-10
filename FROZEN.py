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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_seed()

        model_cfg = {'encoder': encoder, 'dynamics': dynamics} # TODO
        self.model = self.get_model(model_cfg)
        self.model = self.load_model()

    def get_dataloader(self, datapaths, train=True):
        cfg = get_frozen_physion_cfg(debug=self.debug)
        cfg.freeze()
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.IMSIZE,
            seq_len=cfg.SEQ_LEN,
            state_len=cfg.STATE_LEN,
            debug=self.debug,
            train=train,
            )
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=train)
        return dataloader

    def get_model(self, model_cfg):
        model =  modules.FrozenPhysion(model_cfg['encoder'], model_cfg['dynamics']).to(self.device)
        # model = torch.nn.DataParallel(model) # TODO: multi-gpu doesn't work yet, also for loading
        return model

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

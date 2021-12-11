import numpy as np
import logging
import torch

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.data.pydata import TDWDataset
from physion.models.fitvid import FitVid

class FitVidModel(PytorchModel):
    def get_model(self):
        fitvid_params = {
            'input_size': 3,
            'num_channels': 16,
            'g_dim': 32,
            'rnn_size': 64,
            'n_past': self.pretraining_cfg.DATA.STATE_LEN,
            'beta': 1e-4
        }
        model = FitVid(**fitvid_params).to(self.device)
        return model

class PretrainingObjective(FitVidModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def train_step(self, data):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR)
        optimizer.zero_grad()

        loss, _, _, _ = self.model(data['images'].to(self.device)) # train video length = 12
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e2)
        optimizer.step()
        return loss.item()

    def val_step(self, data):
        self.model.train = False
        with torch.no_grad():
            loss, _, _, metrics = self.model(data['images'].to(self.device))
        val_res =  {'val_loss': loss.item()}
        return val_res

class ExtractionObjective(FitVidModel, ExtractionObjectiveBase):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def extract_feat_step(self, data):
        self.model.train = False
        with torch.no_grad():
            loss, preds, h_preds, metrics = self.model(data['images'].to(self.device))
            print(h_preds.shape)

        labels = data['binary_labels'].cpu().numpy()[:,1:] # skip first label to match length of preds -- all the same anyways
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        h_preds = h_preds.cpu().numpy()
        output = {
            'input_states': h_preds[:,:self.model.n_past-1],
            'observed_states': h_preds[:, self.model.n_past-1:], # TODO: hack to pass check feats
            'simulated_states': h_preds[:,self.model.n_past-1:],
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output
            

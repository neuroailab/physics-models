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
            'n_past': 2,
            'beta': 1e-4
        }

        model = FitVid(**fitvid_params)
        return model

class PretrainingObjective(FitVidModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def train_step(self, data):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR)
        optimizer.zero_grad()

        loss, preds, metrics = self.model(data['images']) # train video length = 12
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e2)
        optimizer.step()
        return loss.item()

    def val_step(self, data):
        self.model.train = False
        with torch.no_grad():
            loss, _, _ = self.model(data['images'])

        return {'val_loss': loss.item()}

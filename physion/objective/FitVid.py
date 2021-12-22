import os
import numpy as np
import logging
import imageio
import torch
import mlflow

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.data.pydata import TDWDataset
from physion.models.fitvid import FitVid

BASE_FPS = 30

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

        model_output = self.model(data['images'].to(self.device)) # train video length = 12
        loss = model_output['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e2)
        optimizer.step()
        return loss.item()

    def val_step(self, data):
        self.model.train = False
        with torch.no_grad():
            model_output = self.model(data['images'].to(self.device))
            loss = model_output['loss']
        val_res =  {'val_loss': loss.item()}
        return val_res

def add_border(arr, color=[0,255,0]):
    assert type(arr) == np.ndarray
    assert arr.ndim == 4 
    assert arr.shape[3] == 3, arr.shape

    width = int(0.025 * max(*arr.shape[2:]))
    pad_width = [(0,0), (width, width), (width, width)]
    assert len(color) == 3
    r_cons, g_cons, b_cons = color
    r_, g_, b_ = arr[:, :, :, 0], arr[:, :, :, 1], arr[:, :, :, 2]
    rb = np.pad(array=r_, pad_width=pad_width, mode='constant', constant_values=r_cons)
    gb = np.pad(array=g_, pad_width=pad_width, mode='constant', constant_values=g_cons)
    bb = np.pad(array=b_, pad_width=pad_width, mode='constant', constant_values=b_cons)
    arr = np.stack([rb, gb, bb], axis=-1)
    return arr

def add_rollout_border(arr, rollout_len):
    arr_inp = arr[:-rollout_len]
    arr_inp = add_border(arr_inp, [255, 0, 0])
    arr_pred = arr[-rollout_len:]
    arr_pred = add_border(arr_pred)
    arr = np.concatenate([arr_inp, arr_pred], axis=0)
    return arr
    

class ExtractionObjective(FitVidModel, ExtractionObjectiveBase):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def setup(self):
        super().setup()
        self.count = 0

    def extract_feat_step(self, data):
        self.model.train = False
        with torch.no_grad():
            model_output = self.model(data['images'].to(self.device))
            preds = model_output['preds']
            h_preds = model_output['h_preds'].cpu().numpy()

            # get observed states
            hidden, skips = self.model.encoder(data['images'].to(self.device))
            observed_preds = self.model.decoder(torch.sigmoid(hidden), skips)
            observed_hs = hidden.cpu().numpy()[:,1:] # to match shape of simulated states
            print(observed_hs.shape)

        # save first sample in batch
        rollout_len = self.pretraining_cfg.DATA.SEQ_LEN - self.pretraining_cfg.DATA.STATE_LEN
        fn = os.path.join(self.output_dir, f'gt_{self.count}_'+data['stimulus_name'][0]+'.mp4')
        arr = (255*torch.permute(data['images'][0], (0,2,3,1)).numpy()).astype(np.uint8)
        arr = add_rollout_border(arr, rollout_len)
        imageio.mimwrite(fn, arr, fps=BASE_FPS//self.pretraining_cfg.DATA.SUBSAMPLE_FACTOR, macro_block_size=None)
        mlflow.log_artifact(fn, artifact_path='videos')
        logging.info(f'Video written to {fn}')

        fn = os.path.join(self.output_dir, f'obs_{self.count}_'+data['stimulus_name'][0]+'.mp4')
        arr = (255*torch.permute(observed_preds[0], (0,2,3,1)).cpu().numpy()).astype(np.uint8)
        arr = add_rollout_border(arr, rollout_len)
        imageio.mimwrite(fn, arr, fps=BASE_FPS//self.pretraining_cfg.DATA.SUBSAMPLE_FACTOR, macro_block_size=None)
        mlflow.log_artifact(fn, artifact_path='videos')
        logging.info(f'Video written to {fn}')

        fn = os.path.join(self.output_dir, f'sim_{self.count}_'+data['stimulus_name'][0]+'.mp4')
        arr = (255*torch.permute(preds[0], (0,2,3,1)).cpu().numpy()).astype(np.uint8)
        arr = add_rollout_border(arr, rollout_len)
        imageio.mimwrite(fn, arr, fps=BASE_FPS//self.pretraining_cfg.DATA.SUBSAMPLE_FACTOR, macro_block_size=None)
        mlflow.log_artifact(fn, artifact_path='videos')
        logging.info(f'Video written to {fn}')
        self.count += 1

        labels = data['binary_labels'].cpu().numpy()[:,1:] # skip first label to match length of preds -- all the same anyways
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        output = {
            'input_states': h_preds[:,:self.pretraining_cfg.DATA.STATE_LEN-1],
            'observed_states': observed_hs[:, self.pretraining_cfg.DATA.STATE_LEN-1:],
            'simulated_states': h_preds[:,self.pretraining_cfg.DATA.STATE_LEN-1:],
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output
            

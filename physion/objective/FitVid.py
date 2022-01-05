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
        model = FitVid(
            input_size=3, 
            n_past=self.pretraining_cfg.DATA.STATE_LEN, 
            **self.pretraining_cfg.MODEL
            ).to(self.device)
        return model

class PretrainingObjective(FitVidModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def setup(self):
        super().setup()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR)
        self.optimizer.zero_grad()

    def train_step(self, data):
        self.model.train()

        model_output = self.model(data['images'].to(self.device)) # train video length = 12
        loss = model_output['loss']
        loss = loss / self.pretraining_cfg.TRAIN.ACCUMULATION_STEPS # normalize loss since using average
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e2)
        if self.step % self.pretraining_cfg.TRAIN.ACCUMULATION_STEPS == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def val_step(self, data):
        self.model.training = False
        self.model.eval()
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
        self.model.training = False
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(data['images'].to(self.device))
            preds = model_output['preds']
            h_preds = model_output['h_preds'].cpu().numpy()

            # get observed states
            video = data['images'].to(self.device)
            self.model.B, self.model.T = video.shape[:2]
            pred_s = self.model.frame_predictor.init_states(self.model.B, video.device)
            prior_s = self.model.prior.init_states(self.model.B, video.device)
            hidden, skips = self.model.encoder(data['images'].to(self.device))
            observed_preds  = []
            observed_h_preds = []
            for t in range(self.model.T):
                h = hidden[:, t]
                s = self.model._broadcast_context_frame_skips(skips, frame=t, num_times=1)
                prior_s, (z_t, _, _) = self.model.prior(h, prior_s)
                inp = self.model.get_input(h, None, z_t)
                pred_s, (_, h_pred, _) = self.model.frame_predictor(inp, pred_s)
                h_pred = torch.sigmoid(h_pred)
                x_pred = self.model.decoder(h_pred.unsqueeze(1), s)[:,0]

                observed_h_preds.append(h_pred)
                observed_preds.append(x_pred)

            observed_hs = torch.stack(observed_h_preds, 1).cpu().numpy()
            observed_preds = torch.stack(observed_preds, 1)

        # save N samples in batch
        N = min(data['images'].shape[0], 1)
        for i in range(N):
            rollout_len = self.pretraining_cfg.DATA.SEQ_LEN - self.pretraining_cfg.DATA.STATE_LEN
            fn = os.path.join(self.output_dir, f'gt_{self.count}_'+data['stimulus_name'][i]+'.mp4')
            arr = (255*torch.permute(data['images'][i], (0,2,3,1)).numpy()).astype(np.uint8)
            arr = add_rollout_border(arr, rollout_len)
            imageio.mimwrite(fn, arr, fps=BASE_FPS//self.pretraining_cfg.DATA.SUBSAMPLE_FACTOR, macro_block_size=None)
            mlflow.log_artifact(fn, artifact_path='videos')
            logging.info(f'Video written to {fn}')

            fn = os.path.join(self.output_dir, f'obs_{self.count}_'+data['stimulus_name'][i]+'.mp4')
            arr = (255*torch.permute(observed_preds[i], (0,2,3,1)).cpu().numpy()).astype(np.uint8)
            arr = add_rollout_border(arr, rollout_len)
            imageio.mimwrite(fn, arr, fps=BASE_FPS//self.pretraining_cfg.DATA.SUBSAMPLE_FACTOR, macro_block_size=None)
            mlflow.log_artifact(fn, artifact_path='videos')
            logging.info(f'Video written to {fn}')

            fn = os.path.join(self.output_dir, f'sim_{self.count}_'+data['stimulus_name'][i]+'.mp4')
            arr = (255*torch.permute(preds[i], (0,2,3,1)).cpu().numpy()).astype(np.uint8)
            arr = add_rollout_border(arr, rollout_len)
            imageio.mimwrite(fn, arr, fps=BASE_FPS//self.pretraining_cfg.DATA.SUBSAMPLE_FACTOR, macro_block_size=None)
            mlflow.log_artifact(fn, artifact_path='videos')
            logging.info(f'Video written to {fn}')
            self.count += 1

        labels = data['binary_labels'].cpu().numpy()[:,1:] # skip first label to match length of preds -- all the same anyways
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        output = {
            'input_states': h_preds[:,:-rollout_len],
            'observed_states': observed_hs[:,-rollout_len:], 
            'simulated_states': h_preds[:,-rollout_len:],
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output
            

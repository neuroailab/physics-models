import os
import numpy as np
import logging
import imageio
import torch
import mlflow
from skimage.metrics import structural_similarity
from lpips import LPIPS
from  physion.frechet_video_distance import fvd as tf_fvd

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.data.pydata import TDWDataset
from physion.models.fitvid import FitVid

N_VIS_PER_BATCH = 1
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
        self.count = 0

    def train_step(self, data):
        self.model.train()

        model_output = self.model(data['images'].to(self.device)) # train video length = 12
        loss = model_output['loss']
        assert self.pretraining_cfg.TRAIN.ACCUMULATION_BATCH_SIZE % self.pretraining_cfg.BATCH_SIZE == 0, \
            f'accumulation batch size ({self.pretraining_cfg.TRAIN.ACCUMULATION_BATCH_SIZE}) not divisible by batch size ({self.pretraining_cfg.BATCH_SIZE})'
        accumulation_steps = self.pretraining_cfg.TRAIN.ACCUMULATION_BATCH_SIZE // self.pretraining_cfg.BATCH_SIZE
        loss = loss / accumulation_steps # normalize loss since using average
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e2)
        if self.step % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.count % self.pretraining_cfg.LOG_FREQ == 0: # TODO: change to using step intead of count
            # save visualizations
            frames = {
                'gt': data['images'],
                'sim': model_output['preds'].cpu().detach(),
                'stimulus_name': data['stimulus_name'],
                }
            self.count = save_vis(frames, self.pretraining_cfg, self.output_dir, self.count, 'videos/train')
        return loss.item()

    def val_step(self, data):
        self.model.training = False
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(data['images'].to(self.device))
            loss = model_output['loss']
            out_video = model_output['preds'][:, self.model.n_past-1:].cpu().numpy()
            gt = data['images'][:, self.model.n_past:].numpy()

        val_res =  {'val_loss': loss.item()}
        val_res['psnr'] = psnr(gt, out_video, max_val=1.)
        val_res['ssim'] = ssim(gt, out_video, max_val=1.)
        val_res['lpips'] = lpips(gt, out_video)
        val_res['fvd'] = fvd(gt, out_video)

        # save visualizations
        frames = {
            'gt': data['images'],
            'sim': model_output['preds'].cpu(),
            'stimulus_name': data['stimulus_name'],
            }
        self.count = save_vis(frames, self.pretraining_cfg, self.output_dir, self.count, 'videos/val')

        return val_res

def preprocess_video(video, permute=True, merge=True):
    if permute:
        video =  np.transpose(video, (0,1,3,4,2)) # put channels last
    if merge:
        video = np.reshape(video, (-1,) + video.shape[2:]).astype(np.float32)
    return video

def psnr(video_1, video_2, max_val):
    video_1 = preprocess_video(video_1)
    video_2 = preprocess_video(video_2)
    assert video_1.shape == video_2.shape
    mse = np.mean(np.square(video_1 - video_2), axis=(-3,-2,-1))
    psnr_val = np.subtract(
            20 * np.log(max_val) / np.log(10.0),
            np.float32(10 / np.log(10)) * np.log(mse))
    return np.mean(psnr_val).tolist()

def ssim(video_1, video_2, max_val):
    video_1 = preprocess_video(video_1)
    video_2 = preprocess_video(video_2)
    assert video_1.shape == video_2.shape
    dist = np.array([structural_similarity(video_1[i], video_2[i], data_range=max_val, channel_axis=2) for i in range(len(video_1))])
    return np.mean(dist).tolist()

def lpips(video_1, video_2):
    with torch.no_grad():
        video_1 = 2 * torch.from_numpy(preprocess_video(video_1, permute=False)) - 1 # normalize [-1,1]
        video_2 = 2 * torch.from_numpy(preprocess_video(video_2, permute=False)) - 1 # normalize [-1,1]
        assert video_1.shape == video_2.shape
        loss_fn_alex = LPIPS(net='alex') # best forward scores
        dist = loss_fn_alex(video_1, video_2)
        return np.mean(dist.numpy()).tolist()

def fvd(video_1, video_2):
    video_1 = preprocess_video(video_1, merge=False)
    video_2 = preprocess_video(video_2, merge=False)
    return tf_fvd(video_1, video_2)

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
            out_video = model_output['preds'][:, self.model.n_past-1:].cpu().numpy()
            gt = data['images'][:, self.model.n_past:].numpy()

            # get observed states
            video = data['images'].to(self.device)
            self.model.B, self.model.T = video.shape[:2]
            pred_s = self.model.frame_predictor.init_states(self.model.B, video.device)
            prior_s = self.model.prior.init_states(self.model.B, video.device)
            hidden, skips = self.model.encoder(data['images'].to(self.device))
            observed_preds  = []
            observed_h_preds = []
            for t in range(self.model.T):
                h_pred = hidden[:, t]
                s = self.model._broadcast_context_frame_skips(skips, frame=t, num_times=1)
                # h_pred = torch.sigmoid(h_pred)
                x_pred = self.model.decoder(h_pred.unsqueeze(1), s)[:,0]

                observed_h_preds.append(h_pred)
                observed_preds.append(x_pred)

            observed_hs = torch.stack(observed_h_preds, 1).cpu().numpy()
            observed_preds = torch.stack(observed_preds, 1)

        val_res = {}
        val_res['psnr'] = psnr(gt, out_video, max_val=1.)
        val_res['ssim'] = ssim(gt, out_video, max_val=1.)
        val_res['lpips'] = lpips(gt, out_video)
        val_res['fvd'] = fvd(gt, out_video)
        mlflow.log_metrics(val_res) # TODO: use batch index as step

        # save visualizations
        frames = {
            'gt': data['images'],
            'obs': observed_preds.cpu(),
            'sim': preds.cpu(),
            'stimulus_name': data['stimulus_name'],
            }
        self.count = save_vis(frames, self.pretraining_cfg, self.output_dir, self.count)

        labels = data['binary_labels'].cpu().numpy()[:,1:] # skip first label to match length of preds -- all the same anyways
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        rollout_len = self.pretraining_cfg.DATA.SEQ_LEN - self.pretraining_cfg.DATA.STATE_LEN
        output = {
            'input_states': h_preds[:,:-rollout_len],
            'observed_states': observed_hs[:,-rollout_len:], 
            'simulated_states': h_preds[:,-rollout_len:],
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output
            
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

def save_vis(frames, pretraining_cfg, output_dir, count=0, artifact_path='videos'):
    rollout_len = pretraining_cfg.DATA.SEQ_LEN - pretraining_cfg.DATA.STATE_LEN
    fps = BASE_FPS // pretraining_cfg.DATA.SUBSAMPLE_FACTOR
    n_vis_per_batch = min(pretraining_cfg.BATCH_SIZE, N_VIS_PER_BATCH)
    stimulus_name = frames.pop('stimulus_name')
    for i in range(n_vis_per_batch):
        for k,v in frames.items():
            fn = os.path.join(output_dir, f'{count:06}_'+stimulus_name[i]+f'_{k}.mp4')
            arr = (255*torch.permute(v[i], (0,2,3,1)).numpy()).astype(np.uint8)
            arr = add_rollout_border(arr, rollout_len)
            imageio.mimwrite(fn, arr, fps=fps, macro_block_size=None)
            mlflow.log_artifact(fn, artifact_path=artifact_path)
            logging.info(f'Video written to {fn}')
        count += 1
    return count

import logging
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F

from physopt.objective.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME
from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.data.pydata import TDWDatasetBase

from neuralphys.models.rpin import Net
from neuralphys.utils.misc import tprint

from scipy.spatial.transform import Rotation as R
import os
import io
import glob
import h5py
import json
from PIL import Image
from torchvision import transforms

class RPINModel(PytorchModel):
    def get_model(self):
        model = Net(self.pretraining_cfg.MODEL)
        return model.to(self.device)

class TDWDataset(TDWDatasetBase):
    def __getitem__(self, index):
        with h5py.File(self.hdf5_files[index], 'r') as f: # load ith hdf5 file from list
            frames = list(f['frames'])
            target_contacted_zone = False
            for frame in reversed(frames):
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                if lbl: # as long as one frame touching, label is True
                    target_contacted_zone = True
                    break

            assert len(frames)//self.subsample_factor >= self.seq_len, 'Images must be at least len {}, but are {}'.format(self.seq_len, len(frames)//self.subsample_factor)
            if self.random_seq: # randomly sample sequence of seq_len
                start_idx = self.rng.randint(len(frames)-(self.seq_len*self.subsample_factor)+1)
            else: # get first seq_len # of frames
                start_idx = 0
            end_idx = start_idx + (self.seq_len*self.subsample_factor)
            images = []
            img_transforms = transforms.Compose([
                transforms.Resize((self.imsize, self.imsize)),
                transforms.ToTensor(),
                ])
            bboxes = []
            object_ids = np.array(f['static']['object_ids'])
            for frame in frames[start_idx:end_idx:self.subsample_factor]:
                img = f['frames'][frame]['images']['_img'][()]
                img = Image.open(io.BytesIO(img)) # (256, 256, 3)
                img = img_transforms(img)
                images.append(img)
                xyxys = []
                # print(len(object_ids), object_ids)
                # print([k for k in f['static']['mesh'].keys() if 'vertices' in k])
                for i, obj_id in enumerate(object_ids):
                    obj_id = i # TODO
                    vertices_orig, faces_orig = get_vertices_scaled(f, obj_id)
                    if len(vertices_orig) == 0 or len(faces_orig) == 0: # TODO
                        continue
                    all_pts, all_edges, all_faces = get_full_bbox(vertices_orig)
                    frame_pts = get_transformed_pts(f, all_pts, frame, obj_id)
                    xyxys.append(compute_bboxes(frame_pts, f))
                bboxes.append(xyxys)

            rois = np.array(bboxes, dtype=np.float32)
            num_objs = rois.shape[1]
            max_objs = 15 # self.pretraining_cfg.MODEL.RPIN.NUM_OBJS # TODO
            assert num_objs <= max_objs, f'num objs {num_objs} greater than max objs {max_objs}'
            ignore_mask = np.ones(max_objs, dtype=np.float32)
            if num_objs < max_objs:
                rois = np.pad(rois, [(0,0), (0, max_objs-num_objs), (0,0)])
                ignore_mask[num_objs:] = 0
            rois = torch.from_numpy(rois)
            images = torch.stack(images, dim=0)
            labels = torch.ones((self.seq_len, 1)) if target_contacted_zone else torch.zeros((self.seq_len, 1)) # Get single label over whole sequence
            stimulus_name = f['static']['stimulus_name'][()]

        sample = {
            'data': images[:self.state_len],
            'rois': rois,
            'labels': rois[self.state_len:],
            'data_last': images[:self.state_len],
            'ignore_mask': torch.from_numpy(ignore_mask),
            'stimulus_name': stimulus_name,
            'binary_labels': labels,
            'images': images,
        }
        return sample

class ExtractionObjective(RPINModel, ExtractionObjectiveBase):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, 0)

    def extract_feat_step(self, data):
        self.model.eval()
        with torch.no_grad():
            labels = data['binary_labels'].cpu().numpy()
            stimulus_name = np.array(data['stimulus_name'], dtype=object)
            images, boxes, data_last, ignore_idx = [data[k] for k in ['images', 'rois', 'data_last', 'ignore_mask']]
            images = images.to(self.device)
            rois, coor_features = init_rois(boxes, images.shape)
            rois = rois.to(self.device)
            coor_features = coor_features.to(self.device)
            ignore_idx = ignore_idx.to(self.device)
            outputs = self.model(images, rois, coor_features, num_rollouts=self.pretraining_cfg.MODEL.RPIN.PRED_SIZE_TEST,
                                 data_pred=data_last, phase='test', ignore_idx=ignore_idx)
        input_states = torch.flatten(outputs['input_states'], 2).cpu().numpy()
        observed_states = torch.flatten(outputs['encoded_states'], 2).cpu().numpy()
        simulated_states = torch.flatten(outputs['rollout_states'], 2).cpu().numpy()
            
        output = {
            'input_states': input_states,
            'observed_states': observed_states,
            'simulated_states': simulated_states,
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        # print([(k,v.shape) for k,v in output.items()])
        return output

class PretrainingObjective(RPINModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, 0)

    def train_step(self, data):
        self.model.train() # set to train mode
        self._init_loss()
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.pretraining_cfg.TRAIN.LR,
            weight_decay=self.pretraining_cfg.TRAIN.WEIGHT_DECAY,
        )
        
        # self._adjust_learning_rate() # TODO
        data, boxes, labels, data_last, ignore_idx = [data[k] for k in ['data', 'rois', 'labels', 'data_last', 'ignore_mask']]

        data = data.to(self.device)
        labels = labels.to(self.device)
        rois, coor_features = init_rois(boxes, data.shape)
        rois = rois.to(self.device)
        coor_features = coor_features.to(self.device)
        ignore_idx = ignore_idx.to(self.device)
        # print(data.dtype, labels.dtype, rois.dtype, coor_features.dtype, ignore_idx.dtype)
        optim.zero_grad()
        outputs = self.model(data, rois, coor_features, num_rollouts=self.ptrain_size,
                             data_pred=data_last, phase='train', ignore_idx=ignore_idx)
        loss = self.loss(outputs, labels, 'train', ignore_idx)
        loss.backward()
        optim.step()

        return loss.item() # scalar loss value for the step

    def val_step(self, data):
        with torch.no_grad():
            self.model.eval() # set to eval mode
            self._init_loss()
            data, boxes, labels, data_last, ignore_idx = [data[k] for k in ['data', 'rois', 'labels', 'data_last', 'ignore_mask']]

            data = data.to(self.device)
            labels = labels.to(self.device)
            rois, coor_features = init_rois(boxes, data.shape)
            rois = rois.to(self.device)
            coor_features = coor_features.to(self.device)
            ignore_idx = ignore_idx.to(self.device)
            outputs = self.model(data, rois, coor_features, num_rollouts=self.ptest_size,
                                 data_pred=data_last, phase='test', ignore_idx=ignore_idx)
            loss = self.loss(outputs, labels, 'test', ignore_idx)

        val_res = {
            'val_loss': loss.item(),
        }
        return val_res

    def _init_loss(self):
        cfg = self.pretraining_cfg.MODEL.RPIN
        self.loss_name = []
        self.offset_loss_weight = cfg.OFFSET_LOSS_WEIGHT
        self.position_loss_weight = cfg.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 'o_1', 'o_2']
        if cfg.VAE:
            self.loss_name += ['k_l']
        self.ptrain_size = cfg.PRED_SIZE_TRAIN
        self.ptest_size = cfg.PRED_SIZE_TEST
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.pos_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.off_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation

    def loss(self, outputs, labels, phase='train', ignore_idx=None): # TODO: just pass bbox_rollouts instead of full output?
        C = self.pretraining_cfg.MODEL
        valid_length = self.ptrain_size if phase == 'train' else self.ptest_size

        bbox_rollouts = outputs['bbox']
        # of shape (batch, time, #obj, 4)
        # print(bbox_rollouts.shape, labels.shape)
        loss = (bbox_rollouts - labels) ** 2
        # take mean except time axis, time axis is used for diagnosis
        ignore_idx = ignore_idx[:, None, :, None].to('cuda')
        loss = loss * ignore_idx
        loss = loss.sum(2) / ignore_idx.sum(2)
        loss[..., 0:2] = loss[..., 0:2] * self.offset_loss_weight
        loss[..., 2:4] = loss[..., 2:4] * self.position_loss_weight
        o_loss = loss[..., 0:2]  # offset
        p_loss = loss[..., 2:4]  # position

        for i in range(valid_length):
            self.pos_step_losses[i] += p_loss[:, i].sum(0).sum(-1).mean().item()
            self.off_step_losses[i] += o_loss[:, i].sum(0).sum(-1).mean().item()

        p1_loss = self.pos_step_losses[:self.ptrain_size]
        p2_loss = self.pos_step_losses[self.ptrain_size:]
        self.losses['p_1'] = np.mean(p1_loss)
        self.losses['p_2'] = np.mean(p2_loss)

        o1_loss = self.off_step_losses[:self.ptrain_size]
        o2_loss = self.off_step_losses[self.ptrain_size:]
        self.losses['o_1'] = np.mean(o1_loss)
        self.losses['o_2'] = np.mean(o2_loss)

        # no need to do precise batch statistics, just do mean for backward gradient
        loss = loss.mean(0)
        pred_length = loss.shape[0]
        init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.step / self.pretraining_cfg.TRAIN_STEPS) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(pred_length, out=torch.FloatTensor()))[:, None]
        # tau = torch.cat([torch.ones(self.cons_size, 1), tau], dim=0).to('cuda')
        tau = tau.to(self.device)
        loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()

        if C.RPIN.VAE and phase == 'train':
            kl_loss = outputs['kl_loss']
            self.losses['k_l'] += kl_loss.sum().item()
            loss += C.RPIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        return loss

def init_rois(boxes, shape):
    batch, time_step, _, height, width = shape
    max_objs = boxes.shape[2]
    # coor features, normalized to [0, 1]
    num_im = batch * time_step
    # noinspection PyArgumentList
    co_f = np.zeros(boxes.shape[:-1] + (2,))
    co_f[..., 0] = torch.mean(boxes[..., [0, 2]], dim=-1).numpy().copy() / width
    co_f[..., 1] = torch.mean(boxes[..., [1, 3]], dim=-1).numpy().copy() / height
    coor_features = torch.from_numpy(co_f.astype(np.float32))
    rois = boxes[:, :time_step]
    batch_rois = np.zeros((num_im, max_objs))
    batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
    # noinspection PyArgumentList
    batch_rois = torch.FloatTensor(batch_rois.reshape((batch, time_step, -1, 1)))
    rois = torch.cat([batch_rois, rois], dim=-1)
    return rois, coor_features

def get_camera_matrix(f):
    projection_matrix =  np.array(f['frames']['0000']['camera_matrices']['projection_matrix']).reshape(4,4)
    camera_matrix =  np.array(f['frames']['0000']['camera_matrices']['camera_matrix']).reshape(4,4)
    return np.matmul(projection_matrix, camera_matrix)

def project_points(points, camera_matrix):
    assert points.ndim == 2
    assert points.shape[1] == 3
    
    points = np.pad(points, [(0,0), (0,1)], constant_values=1).T
    projected_points = np.matmul(camera_matrix, points).T
    projected_points = projected_points / projected_points[:,-1:]
    return projected_points

def compute_bbox_from_projected_pts(points):
    assert points.ndim == 2
    assert points.shape[1] == 4
    x1 = np.min(points[:,0])
    y1 = np.min(-points[:,1]) # flip y
    x2 = np.max(points[:,0])
    y2 = np.max(-points[:,1]) # flip y
    xyxy = rescale_xyxy(np.array([x1, y1, x2, y2]))
    return xyxy

def compute_bboxes(points, f):
    camera_matrix = get_camera_matrix(f)
    ppoints = project_points(points, camera_matrix)
    xyxy = compute_bbox_from_projected_pts(ppoints)
    return xyxy

def rescale_xyxy(xyxy):
    xyxy = np.clip(xyxy, -1, 1) # ensure [-1,1]
    xyxy = (xyxy + 1) / 2 # scale to [0,1]
    return xyxy

def get_vertices_scaled(f, obj_id):
    
    vertices_orig = np.array(f['static']['mesh']['vertices_' + str(obj_id)])

    scales = f["static"]["scale"][:]

    vertices_orig[:,0] *= scales[obj_id, 0]
    vertices_orig[:,1] *= scales[obj_id, 1]
    vertices_orig[:,2] *= scales[obj_id, 2]
    faces_orig = np.array(f['static']['mesh']['faces_' + str(obj_id)])
    
    return vertices_orig, faces_orig

def get_full_bbox(vertices):
    arr1 = vertices.min(0)
    
    arr2 = vertices.max(0)
    
    arr = np.stack([arr1, arr2], 0)
    
    pts = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0) , (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
         
    all_edges = [(0, 1), (1, 2), (2, 3), (3, 0),  (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    
    all_faces = [(0, 1, 2, 3), (4, 5, 6, 7), (2, 3, 6, 7), (1, 0, 4, 5), (1, 2, 6, 5), \
                (0, 4, 7, 3)]    
    
    index = np.arange(3)
    
    all_pts = []
    for pt in pts:
        p1 = arr[pt, index]
        all_pts.append(p1)
    
    all_pts = np.stack(all_pts, 0)
    
    return all_pts, all_edges, all_faces    


def get_transformed_pts(f, pts, frame, obj_id):
    rotations_0 = np.array(f['frames'][frame]['objects']['rotations'][obj_id])
    positions_0 = np.array(f['frames'][frame]['objects']['positions'][obj_id])
    
    rot = R.from_quat(rotations_0).as_matrix()
    trans = positions_0
    transformed_pts = np.matmul(rot, pts.T).T + np.expand_dims(trans, axis=0)
    
    return transformed_pts


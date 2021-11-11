import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.models.particle import GNSRigidH
from physion.data.flexdata import PhysicsFleXDataset, collate_fn

use_gpu = torch.cuda.is_available()
class DPINetModel(PytorchModel):
    def get_model(self):
        args = self.pretraining_cfg.MODEL.args
        model = GNSRigidH(args, residual=True, use_gpu=use_gpu)
        if use_gpu:
            model = model.cuda()
        return model

    def load_model(self, model_file):
        checkpoint = torch.load(model_file)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.model.load_state_dict(torch.load(model_file))
        return self.model

class PretrainingObjective(DPINetModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        args = self.pretraining_cfg.DATA.args
        dataset = PhysicsFleXDataset(datapaths, args, 'train', args.verbose_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.pretraining_cfg.BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn)
        return dataloader

    def train_step(self, data):
        args = self.pretraining_cfg.TRAIN.args
        optimizer = optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR, betas=(args.beta1, 0.999))
        optimizer.zero_grad()
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
        criterionMSE = nn.MSELoss()

        self.model.train()
        attr, state, rels, n_particles, n_shapes, instance_idx, label, phases_dict_current= data
        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
        Rr, Rs, Rr_idxs = [], [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]

            Rr_idxs.append(Rr_idx)
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))

            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        data = [attr, state, Rr, Rs, Ra, Rr_idxs, label]

        with torch.set_grad_enabled(True):
            if use_gpu:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t].cuda())
                    else:
                        data[d] = Variable(data[d].cuda())
            else:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t])
                    else:
                        data[d] = Variable(data[d])

            attr, state, Rr, Rs, Ra, Rr_idxs, label = data

            # st_time = time.time()
            predicted = self.model(
                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict_current, args.verbose_model)

        loss = criterionMSE(predicted, label) / args.forward_times
        loss.backward()
        optimizer.step()
        return loss.item()

    def val_step(self, data):
        args = self.pretraining_cfg.TRAIN.args
        criterionMSE = nn.MSELoss()

        self.model.train(False)
        attr, state, rels, n_particles, n_shapes, instance_idx, label, phases_dict_current= data
        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
        Rr, Rs, Rr_idxs = [], [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]

            Rr_idxs.append(Rr_idx)
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))

            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        data = [attr, state, Rr, Rs, Ra, Rr_idxs, label]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t].cuda())
                    else:
                        data[d] = Variable(data[d].cuda())
            else:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t])
                    else:
                        data[d] = Variable(data[d])

            attr, state, Rr, Rs, Ra, Rr_idxs, label = data

            # st_time = time.time()
            predicted = self.model(
                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict_current, args.verbose_model)

        loss = criterionMSE(predicted, label) / args.forward_times
        return {'val_loss': loss.item()}

class ExtractionObjective(DPINetModel, ExtractionObjectiveBase):
    def get_readout_dataprovider(self, datapaths):
        datasets = {phase: PhysicsFleXDataset(
            args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}

        for phase in ['train', 'valid']:
            datasets[phase].load_data(args.env)

        dataloaders = {x: torch.utils.data.DataLoader(
            datasets[x], batch_size=args.batch_size,
            shuffle=True if x == 'train' else False,
            #num_workers=args.num_workers,
            collate_fn=collate_fn)
            for x in ['train', 'valid']}
        return 

    def extract_feat_step(self, data):
        pass

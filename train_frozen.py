import socket
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import physion.modules.frozen as modules
from physion.data.pydata import TDWDataset
from config import get_frozen_physion_cfg
import pdb

DEBUG = True

def run(dataset):
    cfg = get_frozen_physion_cfg(DEBUG)
    cfg.freeze() 

    device = torch.device('cpu')
    encoder = 'deit'
    dynamics = 'mlp'
    model = modules.FrozenPhysion(encoder, dynamics).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR, momentum=0.9)

    dataset = TDWDataset(
        data_root=datasets,
        seq_len=cfg.SEQ_LEN,
        state_len=cfg.STATE_LEN,
        imsize=cfg.IMSIZE, 
        train=True,
        debug=DEBUG, 
        )
    trainloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    for epoch in range(cfg.EPOCHS):
        running_loss = 0.
        for i, data in enumerate(trainloader):
            images = data['images'].to(device)
            inputs = images[:,:4]
            labels = model.get_encoder_feats(images[:,4])
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(running_loss/(i+1))
        

if __name__ == '__main__':
    if socket.gethostname() == 'node19-ccncluster':
        datasets = ['/data1/eliwang/physion/rigid/collide2_new']
    else:
        datasets = ['/mnt/fs4/mrowca/neurips/images/rigid/collide2_new']
    run(datasets)

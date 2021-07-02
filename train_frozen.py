import argparse
import socket
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import physion.modules.frozen as modules
from physion.data.pydata import TDWDataset
from physion.data.new_pydata import TDWDataset as NewTDWDataset
from config import get_frozen_physion_cfg
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--new', action='store_true')
parser.add_argument('--debug', action='store_true')

def run(args):
    if args.new:
        Dataset = NewTDWDataset
        if socket.gethostname() == 'node19-ccncluster':
            datasets = ['/data1/eliwang/physion/rigid/example_dominoes']
        else:
            datasets = ['/mnt/fs4/dbear/tdw_datasets/example_dominoes']
    else:
        Dataset = TDWDataset
        if socket.gethostname() == 'node19-ccncluster':
            datasets = ['/data1/eliwang/physion/rigid/collide2_new']
        else:
            datasets = ['/mnt/fs4/mrowca/neurips/images/rigid/collide2_new']

    cfg = get_frozen_physion_cfg(args.debug)
    cfg.freeze() 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = 'clip'
    dynamics = 'mlp'
    model = modules.FrozenPhysion(encoder, dynamics).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR, momentum=0.9)

    dataset = Dataset(
        data_root=datasets,
        seq_len=cfg.SEQ_LEN,
        state_len=cfg.STATE_LEN,
        imsize=cfg.IMSIZE, 
        train=True,
        debug=args.debug, 
        )
    trainloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    for epoch in range(cfg.EPOCHS):
        running_loss = 0.
        for i, data in enumerate(trainloader):
            images = data['images'].to(device)
            inputs = images[:,:cfg.STATE_LEN]
            labels = model.get_encoder_feats(images[:,cfg.STATE_LEN])
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(running_loss/(i+1))
        

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)

import argparse
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import physion.modules.frozen as modules
from physion.data.pydata import TDWDataset
from physion.data.config import get_data_cfg
from physion.utils import get_subsets_from_datasets
import pdb

def run():
    datasets = ['/data1/eliwang/physion/rigid/collide2_new']
    subsets = get_subsets_from_datasets(datasets)
    cfg  = get_data_cfg(subsets)
    cfg.freeze() 

    device = torch.device('cpu')
    encoder = 'deit'
    dynamics = 'mlp'
    model = modules.FrozenPhysion(encoder, dynamics).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    dataset = TDWDataset(
        data_root=datasets,
        train=True,
        data_cfg=cfg,
        )
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(5):
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
    run()

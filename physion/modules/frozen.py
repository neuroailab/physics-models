from torchvision.models import vgg16_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# ---Encoders---
# Generates latent representation for an image
class DEIT_pretrained(nn.Module):
    def __init__(self):
        super(DEIT_pretrained, self).__init__()
        self.deit = torch.hub.load('facebookresearch/deit:main',
            'deit_base_patch16_224', pretrained=True)
        self.latent_dim = self.deit.norm.normalized_shape[0] # TODO: assumes layer norm is last layer and shape is int
        self.deit.head = nn.Identity() # hack to remove head
    
    def forward(self, x):
        return self.deit(x)

class VGG16_pretrained(nn.Module):
    def __init__(self):
        super(VGG16_pretrained, self).__init__()
        self.vgg = vgg16_bn(pretrained=True)
        self.vgg.classifier = nn.Sequential(*[self.vgg.classifier[i] for i in [0,1,3]]) # get up to second fc w/o dropout
        self.latent_dim = list(self.vgg.modules())[-1].out_features

    def forward(self, x):
        return self.vgg(x)

# ---Dynamics---
# Given a sequence of latent representations, generates the next latent
class ID(nn.Module):
    def forward(self, x):
        assert isinstance(x, list)
        return x[-1] # just return last embedding

class MLP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.regressor = nn.Sequential(
            nn.Linear(self.latent_dim * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim),
            )

    def forward(self, x):
        feats = torch.cat(x, dim=-1)
        pred = self.regressor(feats)
        return pred

class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024)
        self.regressor = nn.Linear(1024, self.latent_dim)
    
    def forward(self, x):
        feats = torch.stack(x) # (T, Bs, self.latent_dim)
        _, hidden = self.lstm(feats)
        x = torch.squeeze(hidden[0].permute(1, 0, 2), dim=1) # assumes n_layers=1
        x = self.regressor(x)
        return x
        
# ---Frozen Physion---
# Given sequence of images, predicts next latent
class FrozenPhysion(nn.Module):
    def __init__(self, encoder, dynamics):
        super().__init__()

        Encoder = _get_encoder(encoder)
        self.encoder = Encoder()

        Dynamics = _get_dynamics(dynamics)
        self.dynamics = Dynamics(self.encoder.latent_dim)

    def forward(self, x):
        # x is (Bs, T, 3, H, W)
        x = self.get_encoder_feats(x)
        x = self.dynamics(x)
        return x

    def get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with torch.no_grad(): # TODO: best place to put this?
            if x.ndim == 4: # (Bs, 3, H, W)
                feats = self._extract_feats(x)
            else:
                assert x.ndim == 5, 'Expected input to be of shape (Bs, T, 3, H, W)'
                feats = []
                for _x in torch.split(x, 1, dim=1):
                    _x = torch.squeeze(_x, dim=1) # _x is shape (Bs, 1, 3, H, W) => (Bs, 3, H, W) TODO: put this in _extract_feats?
                    feats.append(self._extract_feats(_x))
        return feats

    def _extract_feats(self, x):
        feats =  torch.flatten(self.encoder(x), start_dim=1) # (Bs, -1)
        return feats

# ---Utils---
def _get_encoder(encoder):
    if encoder == 'vgg':
        return VGG16_pretrained
    elif encoder == 'deit':
        return DEIT_pretrained
    else:
        raise NotImplementedError

def _get_dynamics(dynamics):
    if dynamics == 'id':
        return ID
    elif dynamics == 'mlp':
        return MLP
    elif dynamics == 'lstm':
        return LSTM
    else:
        raise NotImplementedError
    
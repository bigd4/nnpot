import torch
import torch.nn as nn
from .distance import atom_distances
from .cutoff import CosineCutoff
import logging


class ANI(nn.Module):
    def __init__(self,):
        super(ANI, self).__init__()
        etas = torch.rand(30) + 0.5
        rss = torch.randn(30)
        cut_fn = CosineCutoff(3.0)
        self.G2 = BehlerG2(etas, rss, cut_fn)
        self.layer = nn.Linear(30, 1)

    def forward(self, inputs):
        inputs['positions'] = torch.matmul(inputs['positions'], inputs['scaling'])
        inputs['cell'] = torch.matmul(inputs['cell'], inputs['scaling'])

        positions = inputs['positions']
        neighbors = inputs['neighbors']
        cell = inputs['cell']
        offsets = inputs['offsets']
        mask = inputs['mask']

        inputs['volume'] = torch.sum(
            cell[:, 0] * torch.cross(cell[:, 1], cell[:, 2]), dim=1, keepdim=True
        )
        rij = atom_distances(positions, neighbors, cell, offsets, mask)
        representation = self.G2(rij, mask)
        energy = torch.sum(self.layer(representation), (1, 2))
        return energy


class BehlerG2(nn.Module):
    def __init__(self, etas, rss, cut_fn):
        super(BehlerG2, self).__init__()
        self.etas = nn.Parameter(etas)
        self.rss = nn.Parameter(rss)
        self.cut_fn = cut_fn

    def forward(self, rij, mask):
        x = -self.etas[None, None, None, :] * \
            (rij[:, :, :, None] - self.rss[None, None, None, :]) ** 2
        cut = self.cut_fn(rij).unsqueeze(-1)
        f = torch.exp(x) * cut * mask.unsqueeze(-1)
        f = torch.sum(f, 2).view(rij.size()[0], rij.size()[1], -1)
        return f

import torch
import torch.nn as nn
from ani.distance import atom_distances


class NonePrior(nn.Module):
    def forward(self, inputs):
        positions = inputs['positions']
        f = torch.zeros(positions.size()[0])
        return f


class RepulsivePrior(nn.Module):
    def __init__(self, r_min=0.1, r_max=2.0):
        super(RepulsivePrior, self).__init__()
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        offsets = inputs['offsets']
        mask = inputs['mask']

        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        f = 1 / distances ** 2
        f[distances < self.r_min] = 0.0
        f[distances > self.r_max] = 0.0
        f = torch.sum(f, (1, 2)) / 2  # every pair is counted twice
        return f

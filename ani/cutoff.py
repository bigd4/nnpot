import torch
import torch.nn as nn
import numpy as np


class CosineCutoff(nn.Module):
    def __init__(self, cutoff=3.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class SmoothCosineCutoff(nn.Module):
    def __init__(self, cutoff_smooth=2.8, cutoff=3.0):
        super(SmoothCosineCutoff, self).__init__()
        self.register_buffer("cutoff_smooth", torch.tensor(cutoff_smooth).float())
        self.register_buffer("cutoff", torch.tensor(cutoff).float())

    def forward(self, distances):
        # Compute values of cutoff function
        phase = (distances.clamp(min=self.cutoff_smooth, max=self.cutoff) - self.cutoff_smooth)/\
                 (self.cutoff - self.cutoff_smooth) * np.pi
        cutoffs = 0.5 * (torch.cos(phase) + 1.0) / distances
        return cutoffs


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff=3.0):
        super(PolynomialCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        # Compute values of cutoff function
        cutoffs = (1.0 - (distances / self.cutoff) ** 2) ** 3
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


def polynomial_cut(n):
    coef = np.zeros(n+2)
    coef[0] = 1
    coef[n] = -n-1
    coef[n+1] = n
    return coef

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

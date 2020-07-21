import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, dimension=1):
        super(RBF, self).__init__()
        self.lengthscales = nn.Parameter(torch.ones(dimension))
        self.variance = nn.Parameter(torch.tensor(1.))

    def K(self, a, b):
        lengthscales = torch.log(1 + torch.exp(self.lengthscales))
        variance = torch.log(1 + torch.exp(self.variance))
        a = a / lengthscales
        b = b / lengthscales
        dist = ((a**2).sum(1).view(-1, 1) + (b**2).sum(1) - 2 * torch.mm(a, b.T))
        return variance * torch.exp(-0.5 * dist)

    def Kdiag(self, a):
        return self.variance.expand(a.size(0))

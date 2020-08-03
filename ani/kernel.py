import torch
import torch.nn as nn
from ani.utils import *


class RBF(nn.Module):
    def __init__(self, dimension=1):
        super(RBF, self).__init__()
        # self.lengthscales = nn.Parameter(torch.ones(dimension))
        # self.variance = nn.Parameter(torch.tensor(1.))

        # self.lengthscales = PositiveParameter(torch.ones(dimension))
        self.lengthscales = BoundParameter(torch.ones(dimension), a=0.01, b=100.)
        self.variance = PositiveParameter(torch.tensor(1.))

    def K(self, a, b):
        # float32 may not have enough accuracy
        a = a.double()
        b = b.double()
        # lengthscales = torch.log(1 + torch.exp(self.lengthscales))
        # variance = torch.log(1 + torch.exp(self.variance))
        # a = a / lengthscales
        # b = b / lengthscales
        a = a / self.lengthscales.get()
        b = b / self.lengthscales.get()
        dist = ((a**2).sum(1).view(-1, 1) + (b**2).sum(1) - 2 * torch.mm(a, b.T))
        kernel = self.variance.get() * torch.exp(-0.5 * dist)
        return kernel.float()

    def Kdiag(self, a):
        return self.variance.get().expand(a.size(0))

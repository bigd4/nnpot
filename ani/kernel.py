from ani.utils import *


class RBF(nn.Module):
    def __init__(self, dimension=1):
        super(RBF, self).__init__()
        self.lengthscales = BoundParameter(torch.ones(dimension), a=0.01, b=100.)
        self.variance = PositiveParameter(torch.tensor(1.))

    def K(self, a, b):
        # float32 may not have enough accuracy
        a = a.double()
        b = b.double()
        a = a / self.lengthscales.get()
        b = b / self.lengthscales.get()
        dist = ((a**2).sum(1).view(-1, 1) + (b**2).sum(1) - 2 * torch.mm(a, b.T))
        kernel = self.variance.get() * torch.exp(-0.5 * dist)
        return kernel.float()

    def Kdiag(self, a):
        return self.variance.get().expand(a.size(0))

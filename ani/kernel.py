from ani.utils import *


class RBF(nn.Module):
    def __init__(self, dimension=1):
        super(RBF, self).__init__()
        self.lengthscales = BoundParameter(torch.ones(dimension), a=0.01, b=100.)
        self.variance = PositiveParameter(torch.tensor(1.))

    def K(self, a, b):
        # float32 may not have enough accuracy
        a = a / self.lengthscales.get()
        b = b / self.lengthscales.get()
        dist = ((a**2).sum(1).view(-1, 1) + (b**2).sum(1) - 2 * torch.mm(a, b.T))
        kernel = self.variance.get() * torch.exp(-0.5 * dist)
        return kernel

    def Kdiag(self, a):
        return self.variance.get().expand(a.size(0))

class Hehe(nn.Module):
    def __init__(self):
        super(Hehe, self).__init__()

    def K(self, a, b):
        kernel = torch.sum(a.unsqueeze(1) * b.unsqueeze(0), 2) ** 2 /\
            torch.sum(a * a, 1).unsqueeze(1) / torch.sum(b * b, 1).unsqueeze(0)
        return kernel

    def Kdiag(self, a):
        return torch.ones(a.size(0))

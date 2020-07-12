import torch
import torch.nn as nn
from ani.dataloader import get_dict, _collate_aseatoms
import numpy as np


class DataSet():
    def __init__(self, environment_provider, representation):
        self.environment_provider = environment_provider
        self.representation = representation
        self.frames = []

    def update_dataset(self, frames):
        self.frames.extend(frames)
        tmp_list = _collate_aseatoms([get_dict(atoms, self.environment_provider) for atoms in frames])
        descriptor = self.representation(tmp_list).detach().numpy()
        descriptor = np.sum(descriptor, 1)
        energies = tmp_list['energy'].unsqueeze(-1).numpy()
        if hasattr(self, 'X_array'):
            self.X_array = np.concatenate((self.X_array, descriptor))
            self.y_array = np.concatenate((self.y_array, energies))
        else:
            self.X_array = descriptor
            self.y_array = energies

        self.mean = np.mean(self.X_array, 0)
        self.std = np.std(self.X_array, 0) + 1e-9


    @property
    def X(self):
        return torch.tensor((self.X_array-self.mean)/self.std)

    @property
    def Y(self):
        return torch.tensor(self.y_array)


class RBF(nn.Module):
    def __init__(self, dimension=1):
        super(RBF, self).__init__()
        self.lengthscales = nn.Parameter(torch.ones(dimension))
        #self.lengthscales = nn.Parameter(torch.tensor(1.))
        self.variance = nn.Parameter(torch.tensor(1.))

    def K(self, a, b):
        lengthscales = torch.log(1 + torch.exp(self.lengthscales))
        variance = torch.log(1 + torch.exp(self.variance))
        a = a / lengthscales
        b = b / lengthscales
        dist = (torch.sum(a**2, 1).view(-1, 1) + torch.sum(b**2, 1) - 2 * torch.mm(a, b.T))
        return variance * torch.exp(-0.5 * dist)

    def Kdiag(self, a):
        return self.variance.expand(a.size(0))


class GPR(nn.Module):
    def __init__(self, representation, kern):
        super(GPR, self).__init__()
        self.representation = representation
        self.kern = kern
        self.lamb = nn.Parameter(torch.tensor(0.1))

    def connect_dataset(self, dataset):
        self.dataset = dataset

    def compute_log_likelihood(self):
        lamb = torch.log(1 + torch.exp(self.lamb))
        K = self.kern.K(self.dataset.X, self.dataset.X) + \
            torch.eye(self.dataset.X.size(0)) * lamb

        L = torch.cholesky(K, upper=False)
        V, _ = torch.solve(self.dataset.Y, L)
        V = V.squeeze(1)

        ret = 0.5 * self.dataset.Y.size(0) * torch.tensor(np.log(2 * np.pi))
        ret += torch.log(torch.diag(L)).sum()
        ret += 0.5 * (V ** 2).sum()

        return ret

    def forward(self, inputs):
        lamb = torch.log(1 + torch.exp(self.lamb))
        Xnew = torch.sum(self.representation(inputs), 1)
        mean = torch.tensor(self.dataset.mean)
        std = torch.tensor(self.dataset.std)
        Xnew = (Xnew - mean) / std
        Kx = self.kern.K(self.dataset.X, Xnew)
        K = self.kern.K(self.dataset.X, self.dataset.X) + \
            torch.eye(self.dataset.X.size(0)) * lamb
        L = torch.cholesky(K, upper=False)
        A, _ = torch.solve(Kx, L)
        V, _ = torch.solve(self.dataset.Y, L)

        fmean = torch.mm(A.t(), V)
        fvar = self.kern.Kdiag(Xnew) - (A ** 2).sum(0)
        fvar = fvar.view(-1, 1)

        return fmean, fvar

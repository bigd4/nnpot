import torch
import torch.nn as nn
from ani.dataloader import convert_frames
import numpy as np

class ANI(nn.Module):
    def __init__(self, representation):
        super(ANI, self).__init__()
        self.representation = representation
        n_descriptor = self.representation.dimension
        self.layer = nn.Linear(n_descriptor, 1)

    def set_statistics(self, mean, std):
        self.representation.mean = torch.tensor(mean).float()
        self.representation.std = torch.tensor(std + 1e-9).float()

    def get_energies(self, inputs):
        representation = self.representation(inputs)
        f = self.layer(representation)
        energies = torch.sum(f, (1, 2))
        return energies

    def get_forces(self, inputs):
        inputs['positions'].requires_grad_()
        energies = self.get_energies(inputs)
        forces = -torch.autograd.grad(
            energies.sum(),
            inputs['positions'],
            create_graph=True,
            retain_graph=True
        )[0]
        return forces

    def get_stresses(self, inputs):
        scaling = torch.eye(3, requires_grad=True).float().\
            repeat((inputs['cell'].shape[0], 1, 1))
        inputs['positions'] = torch.matmul(inputs['positions'], scaling)
        inputs['cell'] = torch.matmul(inputs['cell'], scaling)
        volume = torch.sum(
            inputs['cell'][:, 0] *
            torch.cross(inputs['cell'][:, 1], inputs['cell'][:, 2]),
            dim=1, keepdim=True
        )
        energies = self.get_energies(inputs)
        stresses = torch.autograd.grad(
            energies.sum(),
            scaling,
            create_graph=True,
            retain_graph=True
        )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / volume
        return stresses


#TODO
# 1. only atoms near to target atoms be used
# 2. train descriptor parameters directly in GPR model is not stable

class GPR(nn.Module):
    def __init__(self, representation, kern, environment_provider, standardize=True):
        super(GPR, self).__init__()
        self.representation = representation
        self.kern = kern
        self.lamb = nn.Parameter(torch.tensor(-10.))
        self.frames = []
        self.standardize = standardize
        self.environment_provider = environment_provider

    @property
    def X(self):
        if self.standardize:
            return torch.tensor((self.X_array-self.mean)/self.std)
        else:
            return torch.tensor(self.X_array)

    def update_dataset(self, frames):
        self.frames.extend(frames)
        tmp_list = convert_frames(frames, self.environment_provider)
        descriptor = self.representation(tmp_list).sum(1).detach()
        energies = tmp_list['energy'].unsqueeze(-1)
        if hasattr(self, 'X_array'):
            self.X_array = torch.cat((self.X_array, descriptor))
            self.y = torch.cat((self.y, energies))
        else:
            self.X_array = descriptor
            self.y = energies

        self.mean = torch.mean(self.X_array, 0)
        self.std = torch.std(self.X_array, 0) + 1e-9

    def train(self, epoch):
        optimizer = torch.optim.Adam(self.parameters())
        for i in range(epoch):
            loss = self.compute_log_likelihood()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print(i, ':', loss.item())

        lamb = torch.log(1 + torch.exp(self.lamb))
        K = self.kern.K(self.X, self.X) + torch.eye(self.X.size(0)) * lamb
        self.L = torch.cholesky(K, upper=False)
        self.V, _ = torch.solve(self.y, self.L)

    def compute_log_likelihood(self):
        lamb = torch.log(1 + torch.exp(self.lamb))
        K = self.kern.K(self.X, self.X) + torch.eye(self.X.size(0)) * lamb
        L = torch.cholesky(K, upper=False)
        V, _ = torch.solve(self.y, L)
        V = V.squeeze(1)

        ret = 0.5 * self.y.size(0) * torch.tensor(np.log(2 * np.pi))
        ret += torch.log(torch.diag(L)).sum()
        ret += 0.5 * (V ** 2).sum()
        return ret

    def get_energies(self, inputs, with_std=False):
        Xnew = self.representation(inputs).sum(1)
        Xnew = (Xnew - self.mean) / self.std
        Kx = self.kern.K(self.X, Xnew)
        A, _ = torch.solve(Kx, self.L)
        energies = torch.mm(A.t(), self.V).view(-1)
        if with_std:
            energies_var = self.kern.Kdiag(Xnew) - (A ** 2).sum(0)
            energies_var = torch.sqrt(energies_var).view(-1)
            return energies, energies_var
        else:
            return energies

    def get_forces(self, inputs):
        inputs['positions'].requires_grad_()
        energies = self.get_energies(inputs)
        forces = -torch.autograd.grad(
            energies.sum(),
            inputs['positions'],
            create_graph=True,
            retain_graph=True
        )[0]
        return forces

    def get_stresses(self, inputs):
        scaling = torch.eye(3, requires_grad=True).float().\
            repeat((inputs['cell'].shape[0], 1, 1))
        inputs['positions'] = torch.matmul(inputs['positions'], scaling)
        inputs['cell'] = torch.matmul(inputs['cell'], scaling)
        volume = torch.sum(
            inputs['cell'][:, 0] *
            torch.cross(inputs['cell'][:, 1], inputs['cell'][:, 2]),
            dim=1, keepdim=True
        )
        energies = self.get_energies(inputs)
        stresses = torch.autograd.grad(
            energies.sum(),
            scaling,
            create_graph=True,
            retain_graph=True
        )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / volume
        return stresses

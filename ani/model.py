import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ani.dataloader import convert_frames, AtomsData, _collate_aseatoms
from ani.utils import get_loss, PositiveParameter
from ani.prior import *
import numpy as np


# 'update_dataset' and 'train' are api for Magus
class ANI(nn.Module):
    def __init__(self, representation, environment_provider, train_descriptor=False):
        super(ANI, self).__init__()
        self.representation = representation
        self.dataset = AtomsData([], environment_provider)
        n_descriptor = self.representation.dimension
        self.layer = nn.Linear(n_descriptor, 1)
        nn_parameters, descriptor_parameters = [], []
        for key, value in self.named_parameters():
            if 'representation' in key:
                descriptor_parameters.append(value)
            else:
                nn_parameters.append(value)
        self.nn_optimizer = torch.optim.Adam(nn_parameters)
        self.train_descriptor = train_descriptor
        if train_descriptor:
            self.descriptor_optimizer = torch.optim.Adam(descriptor_parameters)

    def set_statistics(self, mean, std):
        self.representation.mean = torch.tensor(mean).float()
        self.representation.std = torch.tensor(std + 1e-9).float()

    def update_dataset(self, frames):
        self.dataset.extend(frames)

    def train(self, epoch=1000):
        train_loader = DataLoader(self.dataset, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
        for i in range(epoch):
            print(i)
            # optimize descriptor parameters
            if self.train_descriptor:
                if i % 5 == 0:
                    for i_batch, batch_data in enumerate(train_loader):
                        loss = get_loss(self, batch_data)
                        self.descriptor_optimizer.zero_grad()
                        loss.backward()
                        self.descriptor_optimizer.step()

            for i_batch, batch_data in enumerate(train_loader):
                loss = get_loss(self, batch_data)
                self.nn_optimizer.zero_grad()
                loss.backward()
                self.nn_optimizer.step()

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
        inputs['scaling'].requires_grad_()
        volume = torch.sum(
            inputs['cell'][:, 0] *
            torch.cross(inputs['cell'][:, 1], inputs['cell'][:, 2], dim=-1),
            dim=1, keepdim=True
        )
        energies = self.get_energies(inputs)
        stresses = torch.autograd.grad(
            energies.sum(),
            inputs['scaling'],
            create_graph=True,
            retain_graph=True
        )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / volume
        return stresses


#TODO
# 1. only atoms near to target atoms be used
# 2. train descriptor parameters directly in GPR model is not stable

class GPR(nn.Module):
    def __init__(self, representation, kern, environment_provider, prior=None, standardize=True):
        super(GPR, self).__init__()
        self.representation = representation
        self.kern = kern
        self.lamb = PositiveParameter(torch.tensor(-10.))
        self.frames = []
        self.standardize = standardize
        self.environment_provider = environment_provider
        self.prior = prior or NonePrior()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer2 = torch.optim.LBFGS(self.parameters(), lr=1e-3, max_iter=40)

    @property
    def X(self):
        if self.standardize:
            return (self.X_array-self.mean)/self.std
        else:
            return self.X_array

    def update_dataset(self, frames):
        self.frames.extend(frames)
        # not enough memory: Buy new RAM! wdnmd
        # tmp_list = convert_frames(frames, self.environment_provider)
        data = AtomsData(frames, self.environment_provider)
        data_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
        for i_batch, batch_data in enumerate(data_loader):
            descriptor = self.representation(batch_data).sum(1).detach()
            energies = batch_data['energy'].unsqueeze(-1)
            prior_energies = self.prior(batch_data).detach().unsqueeze(-1)

            if hasattr(self, 'X_array'):
                self.X_array = torch.cat((self.X_array, descriptor))
                self.y = torch.cat((self.y, energies - prior_energies))
            else:
                self.X_array = descriptor
                self.y = energies - prior_energies

        self.mean = torch.mean(self.X_array, 0).detach()
        self.std = torch.std(self.X_array, 0).detach() + 1e-9

    def recompute_X_array(self):
        tmp_list = convert_frames(self.frames, self.environment_provider)
        self.X_array = self.representation(tmp_list).sum(1)
        self.mean = torch.mean(self.X_array, 0).detach()
        self.std = torch.std(self.X_array, 0).detach() + 1e-9

    def train(self, epoch):
        # def eval_model():
        #     likelihood = self.compute_log_likelihood()
        #     self.optimizer2.zero_grad()
        #     likelihood.backward()
        #     return likelihood
        #
        # for i in range(epoch):
        #     likelihood = self.compute_log_likelihood()
        #     self.optimizer2.zero_grad()
        #     likelihood.backward()
        #     self.optimizer2.step(eval_model)
        #     if i % 5 == 0:
        #         print(i, ':', likelihood.item())
        for i in range(epoch):
            loss = self.compute_log_likelihood()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 500 == 0:
                print(i, ':', loss.item())
        K = self.kern.K(self.X, self.X) + torch.eye(self.X.size(0)) * self.lamb.get()
        self.L = torch.cholesky(K, upper=False).detach()
        self.V = torch.solve(self.y, self.L)[0].detach()

    def compute_log_likelihood(self):
        K = self.kern.K(self.X, self.X) + torch.eye(self.X.size(0)) * self.lamb.get()
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
        energies = torch.mm(A.t(), self.V).view(-1) + self.prior(inputs)
        if with_std:
            energies_var = self.kern.Kdiag(Xnew) - (A ** 2).sum(0)
            energies_std = torch.sqrt(energies_var).view(-1)
            return energies, energies_std
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
        inputs['scaling'].requires_grad_()
        volume = torch.sum(
            inputs['cell'][:, 0] *
            torch.cross(inputs['cell'][:, 1], inputs['cell'][:, 2], dim=-1),
            dim=1, keepdim=True
        )
        energies = self.get_energies(inputs)
        stresses = torch.autograd.grad(
            energies.sum(),
            inputs['scaling'],
            create_graph=True,
            retain_graph=True
        )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / volume
        return stresses

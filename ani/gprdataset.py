from ani.dataloader import _collate_aseatoms, AtomsData, convert_frames
from torch.utils.data import DataLoader
import torch
from scipy.linalg import block_diag
import numpy as np
from ani.utils import *


def remove_dul(X, tolerance=1e-2):
    remain_indices = []
    dist = (X ** 2).sum(1).view(-1, 1) + (X ** 2).sum(1) - 2 * torch.mm(X, X.T)
    for i in range(X.size()[0]):
        for j in remain_indices:
            if dist[i, j] < tolerance:
                break
        else:
            remain_indices.append(i)
    return X[remain_indices]


class GPRData:
    def __init__(self, environment_provider, representation, prior, standardize=True):
        self.environment_provider = environment_provider
        self.representation = representation
        self.prior = prior
        self.initialize_data()
        self.standardize = standardize

    def initialize_data(self):
        self.frames = []
        self.y = None
        self.X_array = None

    def get_data(self, inputs):
        X = self.representation(inputs)
        X = X.view(-1, X.size()[-1])
        if self.standardize:
            X = (X - self.mean) / self.std
        return X

    def update_data(self, atoms_list):
        self.frames.extend(atoms_list)
        self.update_X(atoms_list)
        self.update_y(atoms_list)
        self.update_mean_and_std()

    def update_X(self, atoms_list):
        for atoms in atoms_list:
            data = convert_frames([atoms], self.environment_provider)
            descriptor = self.representation(data).detach()
            descriptor = descriptor.squeeze(0)
            if self.X_array is None:
                self.X_array = descriptor
            else:
                self.X_array = torch.cat((self.X_array, descriptor))

    def update_y(self, atoms_list):
        for atoms in atoms_list:
            data = convert_frames([atoms], self.environment_provider)
            energies = data['energy'].unsqueeze(-1)
            prior_energies = self.prior(data).detach().unsqueeze(-1)
            if self.y is None:
                self.y = energies - prior_energies
            else:
                self.y = torch.cat((self.y, energies - prior_energies))

    def update_mean_and_std(self):
        self.mean = torch.mean(self.X_array, 0).detach()
        self.std = torch.std(self.X_array, 0).detach() + 1e-9
        self.mean_y = torch.mean(self.y, 0).detach()

    @property
    def X(self):
        if self.standardize:
            return (self.X_array - self.mean) / self.std
        return self.X_array

    @property
    def Y(self):
        return self.y

class SGPRData(GPRData):
    def __init__(self, environment_provider, representation, prior, n_sparse, standardize=True):
        super(SGPRData, self).__init__(environment_provider, representation, prior)
        self.n_sparse = n_sparse

    def initialize_data(self):
        super(SGPRData, self).initialize_data()
        self.L = None

    def update_L(self, atoms_list):
        for atoms in atoms_list:
            if self.L is None:
                self.L = np.ones((len(atoms), 1))
            else:
                self.L = block_diag(self.L, np.ones((len(atoms), 1)))

    def update_data(self, atoms_list):
        super(SGPRData, self).update_data(atoms_list)
        self.update_L(atoms_list)
        self.X_unique = remove_dul(self.X)

    @property
    def X_M(self):
        random_indices = torch.randperm(self.X_unique.size()[0])
        n_sparse = min(self.X_unique.size()[0], self.n_sparse)
        return self.X_unique[random_indices[:n_sparse]]

    @property
    def L_NK(self):
        return torch.tensor(self.L).float()

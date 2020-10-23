import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ani.dataloader import convert_frames, AtomsData, _collate_aseatoms
from ani.utils import *
from ani.prior import *
import numpy as np
from ani.gprdataset import *
import abc


# TODO test whether the speed will be affected if always add dropout(p=0.)
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, activation='shifted_softplus', drop=False, p=0.5):
        super(MLP, self).__init__()
        try:
            self.activation_fn = activation_dict[activation]
        except:
            try:
                self.activation_fn = getattr(torch, activation)
            except:
                raise Exception("Unexpected activation functions")
        self.layer1 = nn.Linear(n_in, n_hidden[0])
        self.layers = nn.ModuleList([nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)])
        self.layer2 = nn.Linear(n_hidden[-1], 1)
        self.dropout = nn.Dropout(p=p) if drop else None

    def forward(self, inputs):
        f = self.get_latent_variables(inputs)
        f = self.layer2(f)
        return f

    def get_latent_variables(self, inputs):
        f = self.layer1(inputs)
        if self.dropout is not None:
            f = self.dropout(f)
        f = self.activation_fn(f)
        for layer in self.layers:
            f = layer(f)
            if self.dropout is not None:
                f = self.dropout(f)
            f = self.activation_fn(f)
        return f


class AtomicModule(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_energies(self, inputs):
        pass

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


# Behler's model, related work: N2P2, PyXtalFF
# ANI-1: an extensible neural network potential with DFT accuracy
# at force field computational cost
# DOI: 10.1039/C6SC05720A
class ANI(AtomicModule):
    def __init__(self, representation, elements, n_hidden=[50, 50], prior=None,
                 mean=0., std=1.,):
        super(ANI, self).__init__()
        self.representation = representation
        self.prior = prior or NonePrior()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.z_Embedding = OneHotEmbedding(elements)
        self.element_net = nn.ModuleList([MLP(self.representation.dimension, n_hidden) for _ in range(len(elements))])

    def set_statics(self, mean, std):
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def get_energies(self, inputs):
        atomic_numbers = inputs['atomic_numbers']
        representation = self.representation(inputs)
        element_energies_set = torch.cat([net(representation) for net in self.element_net], 2)
        element_energies_set = element_energies_set * self.std + self.mean
        element_energies = torch.sum(self.z_Embedding(atomic_numbers) * element_energies_set, 2)
        energies = torch.sum(element_energies, 1) + self.prior(inputs)
        return energies

    def get_latent_variables(self, inputs):
        atomic_numbers = inputs['atomic_numbers']
        representation = self.representation(inputs)
        latent_variables_set = torch.cat([net.get_latent_variables(representation).unsqueeze(2)
                                          for net in self.element_net], 2)
        latent_variables = torch.sum(self.z_Embedding(atomic_numbers)[..., None] * latent_variables_set, dim=(1, 2))
        return latent_variables


# Use dropout to calculate uncertainty
# Uncertainty quantification in molecular simulations with
# dropout neural network potentials
# DOI: 10.1038/s41524-020-00390-8
class DropoutANI(ANI):
    def __init__(self, representation, elements, n_hidden=[50, 50], prior=None,
                 mean=0., std=1., p=0.3):
        super(DropoutANI, self).__init__(representation, elements, n_hidden, prior, mean, std)
        self.element_net = nn.ModuleList(
            [MLP(self.representation.dimension, n_hidden, drop=True, p=p)
             for _ in range(len(elements))])

    def get_energies(self, inputs, with_std=False):
        if with_std:
            all_energies = torch.cat(
                [super(DropoutANI, self).get_energies(inputs).unsqueeze(-1) for _ in range(20)], 1)
            mean_energies = torch.mean(all_energies, 1)
            std_energies = torch.std(all_energies, 1)
            return mean_energies, std_energies
        else:
            energies = super(DropoutANI, self).get_energies(inputs)
            return energies


# to solve long-range charge transfer
# Interatomic potentials for ionic systems with density functional accuracy
# based on charge densities obtained by a neural network
# DOI: 10.1103/PhysRevB.92.045131
# TODO: test cholesky_solve
class GoedeckerNet(AtomicModule):
    def __init__(self, representation, elements, n_hidden=[3, 3], train_descriptor=False):
        super(GoedeckerNet, self).__init__()
        self.representation = representation
        self.z_Embedding = OneHotEmbedding(elements)
        self.element_net = nn.ModuleList()
        for _ in range(len(elements)):
            self.element_net.append(
                nn.Sequential(
                    MLP(self.representation.dimension, n_hidden, 'tanh'),
                    torch.nn.Tanh()))

        self.gaussian_width_Embedding = nn.Embedding(max(elements) + 1, 1)
        self.gaussian_width_Embedding.weight.data = torch.ones((max(elements) + 1, 1))
        self.gaussian_width_Embedding.weight.requires_grad = False
        self.hardness_Embedding = nn.Embedding(max(elements) + 1, 1)
        self.hardness_Embedding.weight.data = torch.ones((max(elements) + 1, 1)) * 0.2
        self.hardness_Embedding.weight.requires_grad = False

    def get_energies(self, inputs):
        nb, na, nd = inputs['positions'].size()
        atomic_numbers = inputs['atomic_numbers']
        positions = inputs['positions']
        all_neighbors = inputs['all_neighbors']
        all_mask = inputs['all_mask']
        q_tot = inputs['q_tot'].unsqueeze(-1)

        representation = self.representation(inputs)
        element_electronegativities_set = torch.cat([net(representation) for net in self.element_net], 2)
        element_electronegativities = torch.sum(self.z_Embedding(atomic_numbers) * element_electronegativities_set, 2)
        A = torch.ones((nb, na + 1, na + 1))
        A[:, -1, -1] = 0.
        alpha_i = self.gaussian_width_Embedding(atomic_numbers).squeeze(-1)
        J_ii = self.hardness_Embedding(atomic_numbers).squeeze(-1)
        gamma_ij = 1 / torch.sqrt(alpha_i[:, :, None] ** 2 + alpha_i[:, None, :] ** 2)
        r_ij = atom_distances(positions, all_neighbors, all_mask, cell=None)
        A_ = torch.erf(gamma_ij * r_ij) / r_ij
        A_[all_mask == 0.] = 0.
        A_diag = J_ii + torch.tensor(np.sqrt(2. / np.pi)) / alpha_i
        A_[:, torch.arange(na), torch.arange(na)] = A_diag
        A[:, :-1, :-1] = A_
        y = torch.cat((element_electronegativities, q_tot), 1).unsqueeze(-1)
        q_i = torch.solve(y, A)[0].squeeze(-1)[:, :-1]
        q_ij = q_i[:, :, None] * q_i[:, None, :]

        first_order_term = torch.sum(element_electronegativities * q_i, 1)
        second_order_term = torch.sum(0.5 * A_diag * q_i ** 2, 1)
        cross_term = torch.sum(torch.triu(q_ij * A_, 1), (1, 2))
        energies = first_order_term + second_order_term + cross_term
        return energies


class NNEnsemble(AtomicModule):
    def __init__(self, models):
        super(NNEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.size = len(models)

    def get_energies(self, inputs, with_std=False):
        all_energies = torch.cat([model.get_energies(inputs).unsqueeze(-1) for model in self.models], 1)
        mean_energies = torch.mean(all_energies, 1)
        if with_std:
            std_energies = torch.std(all_energies, 1)
            return mean_energies, std_energies
        return mean_energies


#TODO
# 1. only atoms near to target atoms be used
# 2. train descriptor parameters directly in GPR model is not stable


class GPR(AtomicModule):
    def __init__(self, representation, kern, environment_provider, prior=None, standardize=True):
        super(GPR, self).__init__()
        self.representation = representation
        self.kern = kern
        self.prior = prior or NonePrior()
        self.data = GPRData(environment_provider, representation, self.prior)
        self.lamb = BoundParameter(torch.tensor(0.01), a=1e-5, b=100.)
        self.frames = []
        self.standardize = standardize
        self.environment_provider = environment_provider
        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer2 = torch.optim.LBFGS(self.parameters(), lr=1e-3, max_iter=40)

    def update_data(self, frames):
        self.data.update_data(frames)

    def train(self, epoch, log_epoch=500):
        for i in range(epoch):
            loss = self.compute_log_likelihood()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % log_epoch == 0:
                print(i, ':', loss.item())
        self.update_para()

    def update_para(self):
        X_N, L_NK = self.data.X, self.data.L_NK
        K = L_NK.t() @ (self.kern.K(X_N, X_N) + torch.eye(X_N.size(0)) * self.lamb.get()) @ L_NK

        self.L = torch.cholesky(K, upper=False).detach()
        self.V = torch.solve(self.data.y, self.L)[0].detach()

    def compute_log_likelihood(self):
        X_N, L_NK = self.data.X, self.data.L_NK
        K = L_NK.t() @ (self.kern.K(X_N, X_N) + torch.eye(X_N.size(0)) * self.lamb.get()) @ L_NK
        L = torch.cholesky(K, upper=False)
        V = torch.solve(self.data.y, L)[0].squeeze(1)

        ret = 0.5 * self.data.y.size(0) * torch.tensor(np.log(2 * np.pi))
        ret += torch.log(torch.diag(L)).sum()
        ret += 0.5 * (V ** 2).sum()
        return ret

    def get_energies(self, inputs, with_std=False):
        nb, na, nd = inputs['positions'].size()
        Xnew = self.data.get_data(inputs)
        Kx = self.data.L_NK.t() @ self.kern.K(self.data.X, Xnew)
        A = torch.solve(Kx, self.L)[0]
        atom_energy = torch.mm(A.t(), self.V).view(nb, na)
        energies = (atom_energy * inputs['atoms_mask']).sum(1) + self.prior(inputs)
        if with_std:
            energies_var = self.kern.Kdiag(Xnew) - (A ** 2).sum(0)
            energies_std = torch.sqrt(energies_var).view(-1)
            return energies, energies_std
        else:
            return energies


class SparseGPR(GPR):
    def __init__(self, representation, kern, environment_provider, prior=None, n_sparse=100, standardize=True):
        super(SparseGPR, self).__init__(representation, kern, environment_provider, prior, standardize)
        self.data = SGPRData(environment_provider, representation, self.prior, n_sparse)
        self.lamb = BoundParameter(torch.tensor(0.01), a=1e-5, b=100.)

    def update_para(self):
        X_N, L_NM, L_NK = self.data.X, self.data.L_NM, self.data.L_NK

        K_N = self.kern.K(X_N, X_N)
        K_M = L_NM.t() @ K_N @ L_NM + self.lamb.get()
        L_M = torch.cholesky(K_M, upper=False)
        K_MK = L_NM.t() @ K_N @ L_NK
        K_K = L_NK.t() @ K_N @ L_NK
        V_MK = torch.solve(K_MK, L_M)[0]

        ell = torch.sqrt(torch.diag(K_K) -
                         torch.sum(V_MK ** 2, 0) + self.lamb.get())

        V_MK /= ell
        Y = self.data.Y.view(-1) / ell
        K_MK /= ell

        A_M = torch.eye(V_MK.size(0)) + V_MK @ V_MK.t()
        a_M = (K_MK @ Y).view(-1, 1)

        L_A = torch.cholesky(A_M, upper=False)
        self.L = (L_M @ L_A).detach()
        self.V = torch.solve(a_M, self.L)[0].detach()
        self.L_NM = L_NM

    def compute_log_likelihood(self):
        X_N, L_NM, L_NK = self.data.X, self.data.L_NM, self.data.L_NK
        K_N = self.kern.K(X_N, X_N)
        K_M = L_NM.t() @ K_N @ L_NM + self.lamb.get()
        L_M = torch.cholesky(K_M, upper=False)
        K_MN = L_NM.t() @ K_N @ L_NK
        K_N = L_NK.t() @ K_N @ L_NK

        V_MN, _ = torch.solve(K_MN, L_M)
        Diag_K = torch.diag(torch.diag(
            K_N - V_MN.t() @ V_MN) + self.lamb.get())
        K = Diag_K + V_MN.t() @ V_MN
        self.tmp_K1 = K

        L = torch.cholesky(K, upper=False)
        V, _ = torch.solve(self.data.Y, L)
        V = V.squeeze(1)

        ret = 0.5 * self.data.Y.size(0) * torch.tensor(np.log(2 * np.pi))
        ret += torch.log(torch.diag(L)).sum()
        ret += 0.5 * (V ** 2).sum()
        return ret

    def get_energies(self, inputs, with_std=False):
        nb, na, nd = inputs['positions'].size()
        Xnew = self.data.get_data(inputs)
        Kx = self.L_NM.t() @ self.kern.K(self.data.X, Xnew)
        A = torch.solve(Kx, self.L)[0]
        atom_energy = torch.mm(A.t(), self.V).view(nb, na)
        energies = (atom_energy * inputs['atoms_mask']).sum(1) + self.prior(inputs)
        if with_std:
            energies_var = self.kern.Kdiag(Xnew) - (A ** 2).sum(0)
            energies_std = torch.sqrt(energies_var).view(-1)
            return energies, energies_std
        else:
            return energies

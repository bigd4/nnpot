import torch
import torch.nn as nn
from ani.distance import atom_distances, triple_distances, neighbor_elements
from ani.utils import *
import numpy as np


class CombinationRepresentation(nn.Module):
    def __init__(self, *functions):
        super(CombinationRepresentation, self).__init__()
        self.functions = nn.ModuleList(functions)
        self.dimension = sum([f.dimension for f in self.functions])
        self.mean = torch.tensor([0.])
        self.std = torch.tensor([1.])

    def forward(self, inputs):
        x = []
        for f in self.functions:
            x.append(f(inputs))
        return (torch.cat(x, dim=2) - self.mean) / self.std


#TODO
# 1. zernike(try another form), SB descriptor and approximate SOAP
# 2. deepmd(need element embedding), smooth deepmd
# element embedding should be changed
#

# Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces
class BehlerG1(nn.Module):
    def __init__(self, n_radius, cut_fn, etas=None, rss=None, train_para=True):
        super(BehlerG1, self).__init__()
        if etas is None:
            etas = torch.rand(n_radius) + 0.5
        else:
            assert len(etas) == n_radius, "length of etas should be same as n_radius"

        if rss is None:
            rss = torch.randn(n_radius)
        else:
            assert len(rss) == n_radius, "length of rss should be same as n_radius"

        if train_para:
            self.etas = nn.Parameter(etas)
            self.rss = nn.Parameter(rss)
            # self.etas = PositiveParameter(etas)
        else:
            self.register_buffer("etas", etas)
            self.register_buffer("rss", rss)
        self.cut_fn = cut_fn
        self.dimension = n_radius

        self.z_Embedding = nn.Embedding(300, 1)
        # self.z_Embedding.weight.data = torch.arange(300)[:, None]
        # self.z_Embedding.weight.requires_grad = False

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        offsets = inputs['offsets']
        mask = inputs['mask']
        atomic_numbers = inputs['atomic_numbers']
        z_ratio = self.z_Embedding(atomic_numbers)
        z_ij = neighbor_elements(z_ratio, neighbors)
        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        x = -self.etas[None, None, None, :] * \
            (distances[:, :, :, None] - self.rss[None, None, None, :]) ** 2
        cut = self.cut_fn(distances).unsqueeze(-1)
        f = torch.exp(x) * cut * mask.unsqueeze(-1)
        # f: (n_batch, n_atoms, n_neigh, n_descriptor)
        # z_ij: (n_batch, n_atoms, n_neigh, n_embedded)
        f = f.unsqueeze(-1) * z_ij.unsqueeze(-2)
        f = torch.sum(f, 2).view(distances.size()[0], distances.size()[1], -1)
        return f


# TODO should train zetas? should distance be saved?
class BehlerG2(nn.Module):
    def __init__(self, n_angular, cut_fn, etas=None, zetas=[1], train_para=True):
        super(BehlerG2, self).__init__()
        if not etas:
            etas = torch.rand(n_angular) + 0.5

        if train_para:
            self.etas = nn.Parameter(etas)
            # self.etas = PositiveParameter(etas)
        else:
            self.register_buffer("etas", etas)

        self.cut_fn = cut_fn
        self.zetas = torch.tensor(zetas)
        self.dimension = len(etas) * 2 * len(zetas)
        self.z_Embedding = nn.Embedding(300, 1)
        self.z_Embedding.weight.data = torch.arange(300)[:, None]
        self.z_Embedding.weight.requires_grad = False

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors_j = inputs['neighbors_j']
        neighbors_k = inputs['neighbors_k']
        mask_triples = inputs['mask_triples']
        offsets = inputs['offsets']
        offsets_j = inputs['offsets_j']
        offsets_k = inputs['offsets_k']
        atomic_numbers = inputs['atomic_numbers']

        z_ratio = self.z_Embedding(atomic_numbers)
        z_ij = neighbor_elements(z_ratio, neighbors_j)
        z_ik = neighbor_elements(z_ratio, neighbors_k)
        z_ijk = z_ij * z_ik

        r_ij, r_ik, r_jk = triple_distances(
            positions, neighbors_j, neighbors_k, offsets_j, offsets_k, cell, offsets, mask_triples)

        x = -self.etas[None, None, None, :] * \
            (r_ij ** 2 + r_ik ** 2 + r_jk ** 2)[..., None]
        cut = self.cut_fn(r_ij) * self.cut_fn(r_ik) * self.cut_fn(r_jk)
        radius_part = torch.exp(x) * cut.unsqueeze(-1)

        cos_theta = (r_ij ** 2 + r_ik ** 2 - r_jk ** 2) / (2.0 * r_ij * r_ik)
        cos_theta[mask_triples == 0.0] = 0.0

        angular_pos = 2 ** (1 - self.zetas[None, None, None, :]) * \
                      ((1.0 - cos_theta[..., None]) ** self.zetas[None, None, None, :])
        angular_neg = 2 ** (1 + self.zetas[None, None, None, :]) * \
                      ((1.0 - cos_theta[..., None]) ** self.zetas[None, None, None, :])

        angular_part = torch.cat((angular_pos, angular_neg), 3)

        angular_part[mask_triples == 0.0] = 0.0
        radius_part[mask_triples == 0.0] = 0.0

        f = radius_part.unsqueeze(-1) * angular_part.unsqueeze(-2)
        # f: (n_batch, n_atoms, n_neigh, n_radius, n_angular)
        # z_ijk: (n_batch, n_atoms, n_neigh, n_embedded)
        f = f.unsqueeze(-1) * z_ijk.unsqueeze(-2).unsqueeze(-2)
        f = torch.sum(f, 2).view(r_ij.size()[0], r_ij.size()[1], -1)
        return f


class BehlerG3(nn.Module):
    def __init__(self, n_angular, cut_fn, etas=None, zetas=[1], train_para=True):
        super(BehlerG3, self).__init__()
        if not etas:
            etas = torch.rand(n_angular) + 0.5

        if train_para:
            self.etas = nn.Parameter(etas)
            # self.etas = PositiveParameter(etas)
        else:
            self.register_buffer("etas", etas)

        self.cut_fn = cut_fn
        self.zetas = torch.tensor(zetas)
        self.dimension = len(etas) * 2 * len(zetas)
        self.z_Embedding = nn.Embedding(300, 1)
        # self.z_Embedding.weight.data = torch.arange(300)[:, None]
        # self.z_Embedding.weight.requires_grad = False

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors_j = inputs['neighbors_j']
        neighbors_k = inputs['neighbors_k']
        mask_triples = inputs['mask_triples']
        offsets = inputs['offsets']
        offsets_j = inputs['offsets_j']
        offsets_k = inputs['offsets_k']
        atomic_numbers = inputs['atomic_numbers']

        z_ratio = self.z_Embedding(atomic_numbers)
        z_ij = neighbor_elements(z_ratio, neighbors_j)
        z_ik = neighbor_elements(z_ratio, neighbors_k)
        z_ijk = z_ij * z_ik

        r_ij, r_ik, r_jk = triple_distances(
            positions, neighbors_j, neighbors_k, offsets_j, offsets_k, cell, offsets, mask_triples)

        x = -self.etas[None, None, None, :] * \
            (r_ij ** 2 + r_ik ** 2)[..., None]
        cut = self.cut_fn(r_ij) * self.cut_fn(r_ik)
        radius_part = torch.exp(x) * cut.unsqueeze(-1)

        cos_theta = (r_ij ** 2 + r_ik ** 2 - r_jk ** 2) / (2.0 * r_ij * r_ik)
        cos_theta[mask_triples == 0.0] = 0.0

        angular_pos = 2 ** (1 - self.zetas[None, None, None, :]) * \
                      ((1.0 - cos_theta[..., None]) ** self.zetas[None, None, None, :])
        angular_neg = 2 ** (1 + self.zetas[None, None, None, :]) * \
                      ((1.0 - cos_theta[..., None]) ** self.zetas[None, None, None, :])

        angular_part = torch.cat((angular_pos, angular_neg), 3)

        angular_part[mask_triples == 0.0] = 0.0
        radius_part[mask_triples == 0.0] = 0.0

        f = radius_part.unsqueeze(-1) * angular_part.unsqueeze(-2)
        # f: (n_batch, n_atoms, n_neigh, n_radius, n_angular)
        # z_ijk: (n_batch, n_atoms, n_neigh, n_embedded)
        f = f.unsqueeze(-1) * z_ijk.unsqueeze(-2).unsqueeze(-2)
        f = torch.sum(f, 2).view(r_ij.size()[0], r_ij.size()[1], -1)
        return f


class Zernike(nn.Module):
    def __init__(self, elements, n_max, l_max=None, diag=False, cutoff=5., n_cut=2):
        super(Zernike, self).__init__()
        zernike_coef = torch.tensor(cut_zernike(n_max, n_cut)).float()
        legendre_coef = torch.tensor(legendre(l_max)).float()
        self.R_nl = Polynomial(zernike_coef)
        self.P_l = Polynomial(legendre_coef)
        self.n1, self.n2, self.l = get_zernike_combination(n_max, l_max, diag)
        self.z_Embedding_i = OneHotEmbedding(elements)
        self.z_Embedding_j = AtomicNumberEmbedding(elements)
        self.dimension = len(self.n1) * len(elements)
        self.cutoff = cutoff

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        neighbors_j = inputs['neighbors_j']
        neighbors_k = inputs['neighbors_k']
        mask = inputs['mask']
        mask_triples = inputs['mask_triples']
        offsets = inputs['offsets']
        offsets_j = inputs['offsets_j']
        offsets_k = inputs['offsets_k']
        atomic_numbers = inputs['atomic_numbers']

        z_i = self.z_Embedding_i(atomic_numbers)

        z_ratio = self.z_Embedding_j(atomic_numbers)
        z_ij = neighbor_elements(z_ratio, neighbors)
        distances = atom_distances(positions, neighbors, cell, offsets, mask)

        radius_part = 2 * self.R_nl(distances/self.cutoff)[:, :, :, self.n1, self.l] * \
                      self.R_nl(distances/self.cutoff)[:, :, :, self.n2, self.l]
        radius_part[mask == 0.0] = 0.0
        angular_part = self.P_l(torch.tensor(1.))[self.l].view(1, 1, 1, -1)
        f1 = radius_part * angular_part
        # f1: (n_batch, n_atoms, n_neigh, n_descriptor)
        # z_ij: (n_batch, n_atoms, n_neigh, 1) for now
        f1 = torch.sum(f1 * z_ij * z_ij, 2)

        z_ratio = self.z_Embedding_j(atomic_numbers)
        z_ij = neighbor_elements(z_ratio, neighbors_j)
        z_ik = neighbor_elements(z_ratio, neighbors_k)
        z_ijk = z_ij * z_ik
        r_ij, r_ik, r_jk = triple_distances(
            positions, neighbors_j, neighbors_k, offsets_j, offsets_k, cell, offsets, mask_triples)

        radius_part1 = self.R_nl(r_ij/self.cutoff)[:, :, :, self.n1, self.l] * \
                       self.R_nl(r_ik/self.cutoff)[:, :, :, self.n2, self.l]
        radius_part2 = self.R_nl(r_ik/self.cutoff)[:, :, :, self.n1, self.l] * \
                       self.R_nl(r_ij/self.cutoff)[:, :, :, self.n2, self.l]

        radius_part = radius_part1 + radius_part2
        radius_part[mask_triples == 0.0] = 0.0 # (nb, na, nn, n, n', l)

        cos_theta = (r_ij ** 2 + r_ik ** 2 - r_jk ** 2) / (2.0 * r_ij * r_ik)
        cos_theta[mask_triples == 0.0] = 0.0

        angular_part = self.P_l(cos_theta)[:, :, :, self.l]
        angular_part[mask_triples == 0.0] = 0.0

        f2 = radius_part * angular_part
        # f2: (n_batch, n_atoms, n_neigh_pair, n_descriptor)
        # z_ijk: (n_batch, n_atoms, n_neigh_pair, 1) for now
        f2 = torch.sum(f2 * z_ijk, 2)

        f = f1 + f2
        # f: (n_batch, n_atoms, n_descriptor)
        # z_i: (n_batch, n_atoms, max_element)
        f = f.unsqueeze(-2) * z_i.unsqueeze(-1)

        f = f.view(r_ij.size()[0], r_ij.size()[1], -1)
        return f


class Deepmd_radius(nn.Module):
    def __init__(self, n_radius, cut_fn):
        super(Deepmd_radius, self).__init__()
        self.cut_fn = cut_fn
        self.dimension = n_radius

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        mask = inputs['mask']
        offsets = inputs['offsets']
        atomic_numbers = inputs['atomic_numbers']

        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        cut = self.cut_fn(distances)
        cut[mask == 0.0] = 0.0
        f = torch.zeros(distances.size()[0], distances.size()[1], self.dimension)
        f[:, :, :distances.size()[2]] = cut
        f = f.sort(dim=-1, descending=True)[0]

        return f

class Deepmd_radius(nn.Module):
    def __init__(self, n_radius, cut_fn):
        super(Deepmd_radius, self).__init__()
        self.cut_fn = cut_fn
        self.dimension = n_radius

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        mask = inputs['mask']
        offsets = inputs['offsets']
        atomic_numbers = inputs['atomic_numbers']

        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        cut = self.cut_fn(distances)
        cut[mask == 0.0] = 0.0
        f = torch.zeros(distances.size()[0], distances.size()[1], self.dimension)
        f[:, :, :distances.size()[2]] = cut
        f = f.sort(dim=-1, descending=True)[0]

        return f


class Deepmd_radius(nn.Module):
    def __init__(self, n_radius, cut_fn):
        super(Deepmd_radius, self).__init__()
        self.cut_fn = cut_fn
        self.dimension = n_radius

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        mask = inputs['mask']
        offsets = inputs['offsets']
        atomic_numbers = inputs['atomic_numbers']

        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        cut = self.cut_fn(distances)
        cut[mask == 0.0] = 0.0
        cut = cut.sort(dim=-1, descending=True)[0]
        f = torch.zeros(distances.size()[0], distances.size()[1], self.dimension)
        f[:, :, :distances.size()[2]] = cut
        return f


class Deepmd_angular(nn.Module):
    def __init__(self, n_angular, cut_fn):
        super(Deepmd_angular, self).__init__()
        self.cut_fn = cut_fn
        self.dimension = n_angular * 3

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        mask = inputs['mask']
        offsets = inputs['offsets']
        atomic_numbers = inputs['atomic_numbers']

        distances, dis_vec = \
            atom_distances(positions, neighbors, cell, offsets, mask, vec=True)
        cut = self.cut_fn(distances) / distances
        cut[mask == 0.0] = 0.0
        sorted_index = cut.sort(dim=-1, descending=True)[1].unsqueeze(-1).expand_as(dis_vec)
        # cut_vec: (nb, na, nn, 3)
        # sorted_index: (nb, na, nn, 3) sort on the dim 2 (nn)
        cut_vec = cut.unsqueeze(-1) * dis_vec
        cut_vec = cut_vec.gather(2, sorted_index)
        cut_vec = cut_vec.view(distances.size()[0], distances.size()[1], -1)
        f = torch.zeros(distances.size()[0], distances.size()[1], self.dimension)
        f[:, :, :cut_vec.size()[2]] = cut_vec
        return f

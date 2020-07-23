import torch
import torch.nn as nn
from ani.distance import atom_distances, triple_distances
from ani.utils import PositiveParameter


#TODO
# 1. zernike
# 2. deepmd

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

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        offsets = inputs['offsets']
        mask = inputs['mask']
        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        x = -self.etas[None, None, None, :] * \
            (distances[:, :, :, None] - self.rss[None, None, None, :]) ** 2
        cut = self.cut_fn(distances).unsqueeze(-1)
        f = torch.exp(x) * cut * mask.unsqueeze(-1)
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

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors_j = inputs['neighbors_j']
        neighbors_k = inputs['neighbors_k']
        mask_triples = inputs['mask_triples']
        offsets = inputs['offsets']
        offsets_j = inputs['offsets_j']
        offsets_k = inputs['offsets_k']
        r_ij, r_ik, r_jk = triple_distances(
            positions, neighbors_j, neighbors_k, offsets_j, offsets_k, cell, offsets, mask_triples)

        x = -self.etas[None, None, None, :] * \
            (r_ij ** 2 + r_ik ** 2 + r_jk ** 2)[..., None]
        cut = self.cut_fn(r_ij) * self.cut_fn(r_ik) * self.cut_fn(r_jk)
        radius_part = torch.exp(x) * cut.unsqueeze(-1)

        cos_theta = (r_ij ** 2 + r_ik ** 2 + r_jk ** 2) / (2.0 * r_ij * r_ik)
        tmp = torch.zeros_like(cos_theta)
        tmp[mask_triples != 0] = cos_theta[mask_triples != 0]
        cos_theta = tmp

        angular_pos = 2 ** (1 - self.zetas[None, None, None, :]) * \
                      ((1.0 - cos_theta[..., None]) ** self.zetas[None, None, None, :])
        angular_neg = 2 ** (1 + self.zetas[None, None, None, :]) * \
                      ((1.0 - cos_theta[..., None]) ** self.zetas[None, None, None, :])

        angular_part = torch.cat((angular_pos, angular_neg), 3)
        f = mask_triples[..., None, None] * radius_part.unsqueeze(-1) * angular_part.unsqueeze(-2)
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

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors_j = inputs['neighbors_j']
        neighbors_k = inputs['neighbors_k']
        mask_triples = inputs['mask_triples']
        offsets = inputs['offsets']
        offsets_j = inputs['offsets_j']
        offsets_k = inputs['offsets_k']
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
        f = torch.sum(f, 2).view(r_ij.size()[0], r_ij.size()[1], -1)
        return f


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

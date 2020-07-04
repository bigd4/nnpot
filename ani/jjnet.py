import torch
import torch.nn as nn
from .distance import atom_distances, triple_distances
from .cutoff import CosineCutoff
import logging


class ANI(nn.Module):
    def __init__(self,n_radius, n_angular):
        super(ANI, self).__init__()
        cut_fn = CosineCutoff(3.0)
        self.representation = Representation(n_radius, n_angular, cut_fn)
        self.layer = nn.Linear(30, 1)

    def forward(self, inputs):
        representation = self.representation(inputs)
        energy = torch.sum(self.layer(representation), (1, 2))
        return energy

class Representation(nn.Module):
    def __init__(self, n_radius, n_angular, cut_fn):
        self.RDF = BehlerG1(n_radius, cut_fn)
        self.ADF = BehlerG1(n_angular, cut_fn)

    def forward(self,inputs):
        inputs['positions'] = torch.matmul(inputs['positions'], inputs['scaling'])
        inputs['cell'] = torch.matmul(inputs['cell'], inputs['scaling'])

        positions = inputs['positions']
        neighbors = inputs['neighbors']
        cell = inputs['cell']
        offsets = inputs['offsets']
        mask = inputs['mask']

        inputs['volume'] = torch.sum(
            cell[:, 0] * torch.cross(cell[:, 1], cell[:, 2]), dim=1, keepdim=True
        )

        neighbors_j = inputs['neighbors_j']
        neighbors_k = inputs['neighbors_k']
        mask_triples = inputs['mask_triples']
        offsets_j = inputs['offsets_j']
        offsets_k = inputs['offsets_k']

        # Compute triple distances
        r_ij = atom_distances(positions, neighbors, cell, offsets, mask)
        radius_representation = self.RDF(r_ij, mask)

        r_ij, r_ik, r_jk = triple_distances(
            positions, neighbors_j, neighbors_k, offsets_j, offsets_k, cell, offsets, mask_triples)
        angular_representation = self.ADF(r_ij, r_ik, r_jk, mask_triples)

        representation = torch.cat(radius_representation, angular_representation)
        return representation


#Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces
class BehlerG1(nn.Module):
    def __init__(self, n_radius, cut_fn, etas=None, rss=None, train_para=True):
        super(BehlerG1, self).__init__()
        if etas:
            assert len(etas) == n_radius , "length of etas should be same as n_radius"
        else:
            etas = torch.rand(n_radius) + 0.5

        if rss:
            assert len(rss) == n_radius , "length of rss should be same as n_radius"
        else:
            rss = torch.randn(30)
        
        if train_para:
            self.etas = nn.Parameter(etas)
            self.rss = nn.Parameter(rss)
        else:
            self.register_buffer("etas", etas)
            self.register_buffer("rss", rss)
        self.cut_fn = cut_fn

    def forward(self, r_ij, mask):
        x = -self.etas[None, None, None, :] * \
            (r_ij[:, :, :, None] - self.rss[None, None, None, :]) ** 2
        cut = self.cut_fn(r_ij).unsqueeze(-1)
        f = torch.exp(x) * cut * mask.unsqueeze(-1)
        f = torch.sum(f, 2).view(r_ij.size()[0], r_ij.size()[1], -1)
        return f

#TODO should train zetas?
class BehlerG2(nn.Module):
    def __init__(self, n_angular, cut_fn, etas=None, zetas=[1], train_para=True):
        super(BehlerG2, self).__init__()
        if not etas:
            etas = torch.rand(n_angular) + 0.5

        if train_para:
            self.etas = nn.Parameter(etas)
        else:
            self.register_buffer("etas", etas)
        
        self.cut_fn = cut_fn
        self.zetas = torch.tensor(zetas)

    def forward(self, r_ij, r_ik, r_jk, mask_triples):

        x = -self.etas[None, None, None, :] * \
            (r_ij ** 2 + r_ik ** 2 + r_jk ** 2)[...,None]
        cut  = self.cut_fn(r_ij) * self.cut_fn(r_ik) * self.cut_fn(r_jk)
        radius_part = torch.exp(x) * cut.unsqueeze(-1)

        cos_theta = (r_ij ** 2 + r_ik ** 2 + r_jk ** 2) / (2.0 * r_ij * r_ik)
        cos_theta[triple_masks == 0] = 0.0

        angular_pos = 2 ** (1 - self.zetas[None, None, None, :]) * \
            ((1.0 - cos_theta[...,:]) ** self.zetas[None, None, None, :])
        angular_neg = 2 ** (1 + self.zetas[None, None, None, :]) * \
            ((1.0 - cos_theta[...,:]) ** self.zetas[None, None, None, :])
        angular_part = torch.cat((angular_pos, angular_neg), 3)

        f = mask_triples[..., None, None]* radius_part.unsqueeze(-1) * angular_part.unsqueeze(-2)
        f = torch.sum(f, 2).view(r_ij.size()[0], r_ij.size()[1], -1)
        return f

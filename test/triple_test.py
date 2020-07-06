from ani.environment import ASEEnvironment
from ani.jjnet import ANI
from ani.cutoff import CosineCutoff
from ani.dataloader import get_dict
from ase.io import read
import torch
from ani.distance import triple_distances


def convert(atoms):
    environment_provider = ASEEnvironment(cutoff)
    data = get_dict(atoms, environment_provider)
    for k, v in data.items():
        data[k] = v.unsqueeze(0)
    return data


cutoff = 3.0
frames = read('dataset.traj', ':')

inputs = convert((frames[0]))
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

r_ij, r_ik, r_jk = triple_distances(
    positions, neighbors_j, neighbors_k, offsets_j, offsets_k, cell, offsets, mask_triples)

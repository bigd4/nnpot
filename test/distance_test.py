import torch
from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms


def atom_distances(positions, neighbors, mask, cell=None, offsets=None, vec=False):
    mask = mask < 0.5
    n_batch, n_atoms, n_nbh = neighbors.size()

    idx_m = torch.arange(n_batch)[:, None, None]
    pos_xyz = positions[idx_m, neighbors]

    if cell is not None:
        offsets = offsets.view(n_batch, -1, 3).bmm(cell)
        offsets = offsets.view(n_batch, n_atoms, n_nbh, 3)
        pos_xyz += offsets

    dist_vec = pos_xyz - positions[:, :, None, :]
    distances = torch.norm(dist_vec, 2, 3)
    distances = distances.masked_fill(mask=mask, value=torch.tensor(0.))

    if vec:
        return distances, dist_vec

    return distances

cutoff = 5.
cut_fn = CosineCutoff(cutoff)

frames = read('big.traj', ':')
# frames = augment_data(frames, n=300)
elements = []
for atoms in frames:
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))

# set cutoff and environment_provider
cutoff = 5.
environment_provider = ASEEnvironment(cutoff)

inputs = convert_frames(frames[:1], environment_provider)
inputs['positions'].requires_grad_()
positions = torch.matmul(inputs['positions'], inputs['scaling'])
cell = torch.matmul(inputs['cell'], inputs['scaling'])
neighbors = inputs['neighbors']
offsets = inputs['offsets']
mask = inputs['mask']
atomic_numbers = inputs['atomic_numbers']


etas = torch.rand(10) + 0.5
rss = torch.randn(10)

distances = atom_distances(positions=positions, neighbors=neighbors, mask=mask, cell=cell, offsets=offsets)
x = -etas[None, None, None, :] * \
    (distances[:, :, :, None] - rss[None, None, None, :]) ** 2
cut = cut_fn(distances).unsqueeze(-1)
f = torch.exp(x) * cut * mask.unsqueeze(-1)

print(torch.autograd.grad(f.sum(), positions))
# f: (n_batch, n_atoms, n_neigh, n_descriptor)
# z_ij: (n_batch, n_atoms, n_neigh, n_embedded)
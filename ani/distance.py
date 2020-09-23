import torch
import math


# TODO add an invisible atom at end may get rid of mask
def atom_distances(positions, neighbors, mask, cell=None, offsets=None, vec=False):
    mask = mask < 0.5
    n_batch, n_atoms, n_nbh = neighbors.size()

    idx_m = torch.arange(n_batch)[:, None, None]
    pos_xyz = positions[idx_m, neighbors]

    if cell is not None and torch.det(cell[0]) > 0:
        offsets = offsets.view(n_batch, -1, 3).bmm(cell)
        offsets = offsets.view(n_batch, n_atoms, n_nbh, 3)
        pos_xyz += offsets

    dist_vec = pos_xyz - positions[:, :, None, :]
    distances = torch.norm(dist_vec, 2, 3)
    distances = distances.masked_fill(mask=mask, value=torch.tensor(0.))

    if vec:
        return distances, dist_vec

    return distances


def triple_distances(
        positions, neighbors_j, neighbors_k, offsets_j=None, offsets_k=None,
        cell=None, offsets=None):
    n_batch, n_atoms, n_nbh, n_dim = offsets.size()
    n_pairs = offsets_j.size()[2]

    idx_m = torch.arange(n_batch)[:, None, None]
    pos_j = positions[idx_m, neighbors_j]
    pos_k = positions[idx_m, neighbors_k]

    if cell is not None and torch.det(cell[0]) > 0:
        offsets = offsets.view(n_batch, -1, n_dim).bmm(cell)
        offsets = offsets.view(-1, n_nbh, n_dim)

        offsets_j = offsets_j.view(-1, n_pairs)
        offsets_k = offsets_k.view(-1, n_pairs)

        idx_offset_m = torch.arange(n_batch * n_atoms)[:, None]
        offsets_j = offsets[idx_offset_m, offsets_j[:]].view(n_batch, n_atoms, n_pairs, n_dim)
        offsets_k = offsets[idx_offset_m, offsets_k[:]].view(n_batch, n_atoms, n_pairs, n_dim)

        pos_j = pos_j + offsets_j
        pos_k = pos_k + offsets_k

    R_ij = pos_j - positions[:, :, None, :]
    R_ik = pos_k - positions[:, :, None, :]
    R_jk = pos_j - pos_k

    # + 1e-9 to avoid division by zero
    r_ij = torch.norm(R_ij, 2, 3) + 1e-9
    r_ik = torch.norm(R_ik, 2, 3) + 1e-9
    r_jk = torch.norm(R_jk, 2, 3) + 1e-9

    return r_ij, r_ik, r_jk


def neighbor_elements(z_ratio, neighbors):
    n_batch = z_ratio.size()[0]
    idx_m = torch.arange(n_batch)[:, None, None]
    neighbor_numbers = z_ratio[idx_m, neighbors]
    return neighbor_numbers

# nb*na*nn*nd
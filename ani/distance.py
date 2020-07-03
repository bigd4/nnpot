import torch


def atom_distances(positions, neighbors, cell, offsets, mask):
    n_batch, n_atoms, n_nbh = neighbors.size()

    offsets = offsets.view(n_batch, -1, 3).bmm(cell)
    offsets = offsets.view(n_batch, n_atoms, n_nbh, 3)

    idx_m = torch.arange(n_batch)[:, None, None]
    pos_xyz = positions[idx_m, neighbors] + offsets

    dist_vec = pos_xyz - positions[:, :, None, :]
    distances = torch.norm(dist_vec, 2, 3)

    tmp_distances = torch.zeros_like(distances)
    tmp_distances[mask != 0] = distances[mask != 0]
    distances = tmp_distances
    return distances

import torch


# TODO add an invisible atom at end may get rid of mask
def atom_distances(positions, neighbors, cell, offsets, mask, vec=False):
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

    if vec:
        return distances, dist_vec

    return distances


def triple_distances(
        positions, neighbors_j, neighbors_k, offsets_j, offsets_k,
        cell, offsets, mask_triples):
    n_batch, n_atoms, n_nbh, _ = offsets.size()
    n_pairs = offsets_j.size()[2]

    offsets = offsets.view(n_batch, -1, 3).bmm(cell)
    offsets = offsets.view(-1, n_nbh, 3)

    offsets_j = offsets_j.view(-1, n_pairs)
    offsets_k = offsets_k.view(-1, n_pairs)

    idx_offset_m = torch.arange(n_batch * n_atoms)[:, None]
    offsets_j = offsets[idx_offset_m, offsets_j[:]].view(n_batch, n_atoms, n_pairs, 3)
    offsets_k = offsets[idx_offset_m, offsets_k[:]].view(n_batch, n_atoms, n_pairs, 3)

    idx_m = torch.arange(n_batch)[:, None, None]
    pos_j = positions[idx_m, neighbors_j] + offsets_j
    pos_k = positions[idx_m, neighbors_k] + offsets_k

    R_ij = pos_j - positions[:, :, None, :]
    R_ik = pos_k - positions[:, :, None, :]
    R_jk = pos_j - pos_k

    # + 1e-9 to avoid division by zero
    r_ij = torch.norm(R_ij, 2, 3)
    tmp = torch.zeros_like(r_ij)
    tmp[mask_triples != 0] = r_ij[mask_triples != 0]
    r_ij = tmp

    r_ik = torch.norm(R_ik, 2, 3)
    tmp = torch.zeros_like(r_ik)
    tmp[mask_triples != 0] = r_ik[mask_triples != 0]
    r_ik = tmp

    r_jk = torch.norm(R_jk, 2, 3)
    tmp = torch.zeros_like(r_jk)
    tmp[mask_triples != 0] = r_jk[mask_triples != 0]
    r_jk = tmp

    return r_ij, r_ik, r_jk


def neighbor_elements(z_ratio, neighbors):
    n_batch = z_ratio.size()[0]
    idx_m = torch.arange(n_batch)[:, None, None]
    neighbor_numbers = z_ratio[idx_m, neighbors]
    return neighbor_numbers

# nb*na*nn*nd
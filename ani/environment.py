from ase.neighborlist import neighbor_list
import numpy as np


class ASEEnvironment:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def get_environment(self, atoms):
        n_atoms = len(atoms)
        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", atoms, self.cutoff, self_interaction=False
        )
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

            mask = np.zeros((n_atoms, n_max_nbh), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((n_atoms, n_max_nbh), dtype=np.float32)
            neighborhood_idx[mask] = idx_j

            offset = np.zeros((n_atoms, n_max_nbh, 3), dtype=np.float32)
            offset[mask] = idx_S
        else:
            neighborhood_idx = -np.ones((n_atoms, 1), dtype=np.float32)
            offset = np.zeros((n_atoms, 1, 3), dtype=np.float32)
            mask = np.zeros((n_atoms, 1), dtype=np.bool)

        return neighborhood_idx, offset, mask

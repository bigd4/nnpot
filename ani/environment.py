from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
import numpy as np
import abc


class AtomsEnvironment(abc.ABC):
    @abc.abstractmethod
    def get_environment(self, atoms):
        pass


class SimpleEnvironmentProvider(AtomsEnvironment):
    def get_environment(self, atoms):
        n_atoms = len(atoms)
        neighborhood_idx = np.array([[i for i in range(n_atoms) if i != j]
                                     for j in range(n_atoms)])
        offsets = np.zeros((n_atoms, n_atoms, 3), dtype=np.float32)
        mask = np.ones((n_atoms, n_atoms), dtype=np.bool)
        return neighborhood_idx, offsets, mask


class ASEEnvironment(AtomsEnvironment):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def get_environment(self, atoms):
        n_atoms = len(atoms)
        nb = NeighborList([0.5 * self.cutoff] * n_atoms,
                           skin=0.0, self_interaction=False, bothways=True,
                           primitive=NewPrimitiveNeighborList)
        nb.update(atoms)
        neighborhood_idx, offsets, nbh = [], [], []
        for i in range(n_atoms):
            idx, offset = nb.get_neighbors(i)
            nbh.append(len(idx))
            neighborhood_idx.append(list(idx))
            offsets.append(offset.tolist())
        mask = [None] * n_atoms
        max_nbh = max(max(nbh), 1)
        for i in range(n_atoms):
            neighborhood_idx[i].extend([-1.] * (max_nbh - nbh[i]))
            offsets[i].extend([[0., 0., 0.]] * (max_nbh - nbh[i]))
            mask[i] = [True] * nbh[i] + [False] * (max_nbh - nbh[i])
        neighborhood_idx = np.array(neighborhood_idx, dtype=np.float32)
        offsets = np.array(offsets, dtype=np.float32)
        mask = np.array(mask, dtype=np.bool)
        return neighborhood_idx, offsets, mask

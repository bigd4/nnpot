from torch.utils.data import Dataset
import torch
import numpy as np
import copy


#TODO
# new method collect atom triples(may convert to gpu)
# rotate atoms for deepmd

def collect_atom_triples(nbh_idx):
    n_atoms, n_neigh = nbh_idx.shape
    if n_neigh == 1:
        return np.zeros((5, n_atoms, 1,))

    # Construct possible permutations
    nbh_idx_j = np.tile(nbh_idx, n_neigh)
    nbh_idx_k = np.repeat(nbh_idx, n_neigh).reshape((n_atoms, -1))

    # Remove same interactions and non unique pairs
    triu_idx_row, triu_idx_col = np.triu_indices(n_neigh, k=1)
    triu_idx_flat = triu_idx_row * n_neigh + triu_idx_col
    nbh_idx_j = nbh_idx_j[:, triu_idx_flat]
    nbh_idx_k = nbh_idx_k[:, triu_idx_flat]

    # Keep track of periodic images
    offset_idx = np.tile(np.arange(n_neigh), (n_atoms, 1))

    # Construct indices for pairs of offsets
    offset_idx_j = np.tile(offset_idx, n_neigh)
    offset_idx_k = np.repeat(offset_idx, n_neigh).reshape((n_atoms, -1))

    # Remove non-unique pairs and diagonal
    offset_idx_j = offset_idx_j[:, triu_idx_flat]
    offset_idx_k = offset_idx_k[:, triu_idx_flat]

    mask_triples = np.ones_like(nbh_idx_j)
    mask_triples[nbh_idx_j < 0] = 0
    mask_triples[nbh_idx_k < 0] = 0

    return nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, mask_triples


class AtomsData(Dataset):
    def __init__(self, frames, environment_provider):
        self.frames = frames
        self.environment_provider = environment_provider
        self.datalist = [{} for _ in range(len(frames))]

    def __len__(self):
        return len(self.frames)

    def extend(self, frames):
        self.frames.extend(frames)
        self.datalist.extend([{} for _ in range(len(frames))])

    def __getitem__(self, i):
        if not self.datalist[i]:
            atoms = self.frames[i]
            self.datalist[i] = get_dict(atoms, self.environment_provider)
        return self.datalist[i]


def _collate_aseatoms(examples):
    properties = examples[0]
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in properties.items()
    }

    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }

    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val
    return batch


def get_dict(atoms, environment_provider):
    neighbors, offsets, mask = environment_provider.get_environment(atoms)
    neighbors_j, neighbors_k, offsets_j, offsets_k, mask_triples = collect_atom_triples(neighbors)
    d = {
        'neighbors': torch.from_numpy(neighbors).long(),
        'neighbors_j': torch.from_numpy(neighbors_j).long(),
        'neighbors_k': torch.from_numpy(neighbors_k).long(),
        'offsets_j': torch.from_numpy(offsets_j).long(),
        'offsets_k': torch.from_numpy(offsets_k).long(),
        'offsets': torch.from_numpy(offsets).float(),
        'mask': torch.from_numpy(mask).float(),
        'mask_triples': torch.from_numpy(mask_triples).float(),
        'positions': torch.tensor(atoms.positions).float(),
        'cell': torch.tensor(atoms.cell[:]).float(),
        'atomic_numbers': torch.tensor(atoms.numbers).long(),
        'scaling': torch.eye(3).float(),
        'atoms_mask': torch.ones(len(atoms)),
        'n_atoms': torch.tensor(len(atoms)).float(),
    }

    for key in ['energy', 'forces', 'stress']:
        if key in atoms.info:
            d[key] = torch.tensor(atoms.info[key]).float()
    return d


def convert_frames(frames, environment_provider):
    return _collate_aseatoms([get_dict(atoms, environment_provider) for atoms in frames])

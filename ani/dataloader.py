from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np


def collect_atom_triples(nbh_idx):
    natoms, nneigh = nbh_idx.shape

    # Construct possible permutations
    nbh_idx_j = np.tile(nbh_idx, nneigh)
    nbh_idx_k = np.repeat(nbh_idx, nneigh).reshape((natoms, -1))

    # Remove same interactions and non unique pairs
    triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
    triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
    nbh_idx_j = nbh_idx_j[:, triu_idx_flat]
    nbh_idx_k = nbh_idx_k[:, triu_idx_flat]

    # Keep track of periodic images
    offset_idx = np.tile(np.arange(nneigh), (natoms, 1))

    # Construct indices for pairs of offsets
    offset_idx_j = np.tile(offset_idx, nneigh)
    offset_idx_k = np.repeat(offset_idx, nneigh).reshape((natoms, -1))

    # Remove non-unique pairs and diagonal
    offset_idx_j = offset_idx_j[:, triu_idx_flat]
    offset_idx_k = offset_idx_k[:, triu_idx_flat]

    mask_triples = np.ones_like(nbh_idx_j)
    mask_triples[nbh_idx_j < 0] = 0
    mask_triples[nbh_idx_k < 0] = 0

    return nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, mask_triples


class AtomsData(Dataset):
    def __init__(self,frames,environment_provider):
        self.frames = frames
        self.environment_provider = environment_provider
        self.datalist = [{} for _ in range(len(frames))]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        if not self.datalist[i]:
            atoms = self.frames[i]
            neighbors, offsets, mask = self.environment_provider.get_environment(atoms)
            neighbors_j, neighbors_k, offsets_j, offsets_k, mask_triples = collect_atom_triples(neighbors)

            self.datalist[i]['neighbors'] = torch.from_numpy(neighbors).long()
            self.datalist[i]['neighbors_j'] = torch.from_numpy(neighbors_j).long()
            self.datalist[i]['neighbors_k'] = torch.from_numpy(neighbors_k).long()
            self.datalist[i]['offsets_j'] = torch.from_numpy(offsets_j).long()
            self.datalist[i]['offsets_k'] = torch.from_numpy(offsets_k).long()
            self.datalist[i]['offsets'] = torch.from_numpy(offsets).float()
            self.datalist[i]['mask'] = torch.from_numpy(mask).float()
            self.datalist[i]['mask_triples'] = torch.from_numpy(mask_triples).float()
            self.datalist[i]['positions'] = torch.from_numpy(atoms.positions).float()
            self.datalist[i]['positions'].requires_grad = True
            self.datalist[i]['cell'] = torch.from_numpy(atoms.cell[:]).float()
            self.datalist[i]['cell'].requires_grad = True
            self.datalist[i]['energy'] = torch.tensor(atoms.info['energy']).float()
            self.datalist[i]['forces'] = torch.tensor(atoms.info['forces']).float()
            self.datalist[i]['stress'] = torch.tensor(atoms.info['stress']).float()
            self.datalist[i]['scaling'] = torch.eye(3, requires_grad=True).float()
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

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from ase.io import read
    from environment import ASEEnvironment
    frames = read('dataset.traj',':')
    environment_provider = ASEEnvironment(3.0)
    data = AtomsData(frames,environment_provider)
    
    dataloader = DataLoader(data, batch_size=8, shuffle=True,collate_fn=_collate_aseatoms)
    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)
        print(batch_data['neighbors'].size())
        print(batch_data['offset'].size())
        print(batch_data['positions'].size())
        print(batch_data['cell'].size())
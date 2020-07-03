from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np


class AtomsData(Dataset):
    def __init__(self,frames,environment_provider):
        self.frames = frames
        self.environment_provider = environment_provider

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        atoms = self.frames[i]
        neighbors, offsets, mask = self.environment_provider.get_environment(atoms)
        d = {}
        d['neighbors'] = torch.from_numpy(neighbors).long()
        d['offsets'] = torch.from_numpy(offsets).float()
        d['mask'] = torch.from_numpy(mask).float()
        d['positions'] = torch.from_numpy(atoms.positions).float()
        d['positions'].requires_grad = True
        d['cell'] = torch.from_numpy(atoms.cell[:]).float()
        d['cell'].requires_grad = True
        d['energy'] = torch.tensor(atoms.info['energy']).float()
        d['forces'] = torch.tensor(atoms.info['forces']).float()
        d['stress'] = torch.tensor(atoms.info['stress']).float()
        d['scaling'] = torch.eye(3, requires_grad=True).float()
        return d


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
import torch
import torch.nn as nn
from ani.distance import atom_distances
from ase.atoms import Atoms
import numpy as np


class NonePrior(nn.Module):
    def forward(self, inputs):
        positions = inputs['positions']
        f = torch.zeros(positions.size()[0])
        return f


class AtomRefPrior(nn.Module):
    def __init__(self, atomref):
        super(AtomRefPrior, self).__init__()
        self.atomref = [0. if i not in atomref.keys() else atomref[i] for i in range(100)]
        weight = [[w] for w in self.atomref]
        self.ref_energy = nn.Embedding.from_pretrained(torch.tensor(weight).float())

    def get_energy(self, atoms):
        return np.sum([self.atomref[i] for i in atoms.get_atomic_numbers()])

    def forward(self, inputs):
        if isinstance(inputs, list):
            energy = np.zeros(len(inputs))
            for i, atoms in enumerate(inputs):
                energy[i] = self.get_energy(atoms)
            return energy
        if isinstance(inputs, Atoms):
            return self.get_energy(inputs)
        if isinstance(inputs, dict):
            return self.ref_energy(inputs['atomic_numbers']).sum(1).squeeze(-1)


class RepulsivePrior(nn.Module):
    def __init__(self, r_min=0.1, r_max=2.0):
        super(RepulsivePrior, self).__init__()
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, inputs):
        positions = inputs['positions']
        cell = inputs['cell']
        neighbors = inputs['neighbors']
        offsets = inputs['offsets']
        mask = inputs['mask']

        distances = atom_distances(positions, neighbors, cell, offsets, mask)
        f = 1 / distances ** 2
        f[distances < self.r_min] = 0.0
        f[distances > self.r_max] = 0.0
        f = torch.sum(f, (1, 2)) / 2  # every pair is counted twice
        return f

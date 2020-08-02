import logging
from ase.io import read
from ani.model import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, convert_frames
from torch.utils.data import DataLoader
import torch
from ani.symmetry_functions import BehlerG1, BehlerG3, CombinationRepresentation
from ani.cutoff import CosineCutoff
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression


device = "cpu"

cutoff = 5.0
n_radius = 30
n_angular = 10
environment_provider = ASEEnvironment(cutoff)
cut_fn = CosineCutoff(cutoff)
rdf = BehlerG1(n_radius, cut_fn)
adf = BehlerG3(n_angular, cut_fn)
representation = CombinationRepresentation(rdf, adf)
representation = CombinationRepresentation(rdf)
# model.load_state_dict(torch.load('parameter-new.pkl'))
frames = read('TiO2.traj', ':')
e_mean = np.mean([atoms.info['energy'] for atoms in frames])
for atoms in frames:
    atoms.info['energy'] -= e_mean
n_split = 130


data = convert_frames(frames, environment_provider)
data['positions'].requires_grad_()
X_energy = representation(data)
nb, na, nd = X_energy.size()
X_forces = torch.zeros(nb, na, 3, nd)
for i in range(nd):
    data = convert_frames(frames, environment_provider)
    data['positions'].requires_grad_()
    X_energy = representation(data)
    X_forces[:, :, :, i] = torch.autograd.grad(
        X_energy[:, :, i].sum(),
        data['positions']
    )[0]

Y_energy = np.array([atoms.info['energy'] for atoms in frames])
Y_forces = np.array([atoms.info['forces'] for atoms in frames]).flatten()

X_energy = X_energy.sum(1).detach().numpy()
X_forces = -X_forces.view(-1, nd).detach().numpy()
X = np.concatenate((X_energy, X_forces), 0)
Y = np.concatenate((Y_energy, Y_forces), 0)

model = LinearRegression()
model.fit(X, Y)
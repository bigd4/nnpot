from ani.environment import ASEEnvironment
from ani.jjnet import ANI
from ani.cutoff import CosineCutoff
from ani.dataloader import get_dict
from ase.io import read
import torch
from torch.utils.data import DataLoader
import numpy as np


def convert(atoms):
    environment_provider = ASEEnvironment(cutoff)
    data = get_dict(atoms, environment_provider)
    for k, v in data.items():
        data[k] = v.unsqueeze(0)
    return data

def numerical_forces(a, dx = 1e-2):
    Natoms, dim = a.positions.shape
    F = np.zeros((Natoms, dim))
    for ia in range(Natoms):
        for idim in range(dim):
            a_up = a.copy()
            a_down = a.copy()
            a_up.positions[ia, idim] += dx / 2
            a_down.positions[ia, idim] -= dx / 2
            E_up = net(convert(a_up))
            E_down = net(convert(a_down))
            F[ia, idim] = -(E_up - E_down) / dx
    return F

def numerical_stress(atoms, dx=1e-2):
    stress = np.zeros((3, 3))
    c = atoms.cell[:]
    for i in range(3):
        for j in range(3):
           a_up = atoms.copy()
           a_down = atoms.copy()
           e = np.eye(3)
           e[i,j] += dx/2
           a_up.set_cell(c@e,True)
           e = np.eye(3)
           e[i,j] -= dx/2
           a_down.set_cell(c@e,True)
           E_up = net(convert(a_up))
           E_down = net(convert(a_down))
           stress[i,j] = (E_up - E_down)/dx
    return stress

frames = read('dataset.traj', ':')
atoms = frames[0]

cutoff = 3.0
n_radius = 10
n_angular = 10

net = ANI(n_radius, n_angular, cutoff)

data = convert(atoms)
predict_energy = net(data)
predict_forces = -torch.autograd.grad(
    predict_energy.sum(),
    data['positions'],
    create_graph=True,
    retain_graph=True
)[0].detach().numpy()
predict_stress = torch.autograd.grad(
    predict_energy.sum(),
    data['scaling'],
    create_graph=True,
    retain_graph=True
)[0].detach().numpy()

predict_forces_ = numerical_forces(atoms)
predict_stress_ = numerical_stress(atoms)
import logging
from ase.io import read
from ani.jjnet import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData
from ani.train import Trainer
from torch.utils.data import DataLoader
import torch



cutoff = 3.0
n_radius = 30
n_angular = 0

net = ANI(n_radius, n_angular, cutoff)
net.load_state_dict(torch.load('parameter.pkl'))
loss_fn = torch.nn.MSELoss()

frames = read('dataset.traj', ':')
n_split = 130
environment_provider = ASEEnvironment(cutoff)
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=_collate_aseatoms)

for i_batch, batch_data in enumerate(test_loader):
    predict_energy = net(batch_data)
    predict_forces = -torch.autograd.grad(
        predict_energy.sum(),
        batch_data['positions'],
        create_graph=True,
        retain_graph=True
    )[0]
    predict_stress = torch.autograd.grad(
        predict_energy.sum(),
        batch_data['scaling'],
        create_graph=True,
        retain_graph=True
    )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / batch_data['volume']

    target_energy = batch_data['energy']
    target_forces = batch_data['forces']
    target_stress = batch_data['stress']
    energy_loss = loss_fn(predict_energy, target_energy)
    force_loss = loss_fn(predict_forces, target_forces)
    stress_loss = loss_fn(predict_stress, target_stress)
    loss = energy_loss + force_loss + stress_loss

    print(loss)


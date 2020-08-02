import logging
from ase.io import read
from ani.model import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData
from torch.utils.data import DataLoader
import torch
from ani.symmetry_functions import BehlerG1, BehlerG3, CombinationRepresentation
from ani.cutoff import CosineCutoff
from ani.utils import get_loss
import torch.nn as nn
import numpy as np


device = "cpu"

cutoff = 5.0
n_radius = 30
n_angular = 10
environment_provider = ASEEnvironment(cutoff)
cut_fn = CosineCutoff(cutoff)
rdf = BehlerG1(n_radius, cut_fn)
adf = BehlerG3(n_angular, cut_fn)
representation = CombinationRepresentation(rdf, adf)
# representation = CombinationRepresentation(rdf)
model = ANI(representation, environment_provider)
# model.load_state_dict(torch.load('parameter-new.pkl'))
frames = read('stress.traj', ':')
e_mean = np.mean([atoms.info['energy'] for atoms in frames])
for atoms in frames:
    atoms.info['energy'] -= e_mean
n_split = 130

train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=_collate_aseatoms)

loss_fn = torch.nn.MSELoss()

nn_parameters, descriptor_parameters = [], []
for key, value in model.named_parameters():
    if 'etas' in key or 'rss' in key:
        descriptor_parameters.append(value)
    else:
        nn_parameters.append(value)

nn_optimizer = torch.optim.Adam(nn_parameters)
descriptor_optimizer = torch.optim.Adam(descriptor_parameters)
optimizer = torch.optim.Adam(model.parameters())
epoch = 1000
min_loss = 1000
for i in range(epoch):
    # optimize descriptor parameters
    # if i % 50 == 0:
    #     for i_batch, batch_data in enumerate(train_loader):
    #         batch_data = {k: v.to(device) for k, v in batch_data.items()}
    #         loss = get_loss(model, loss_fn, batch_data)
    #         descriptor_optimizer.zero_grad()
    #         loss.backward()
    #         descriptor_optimizer.step()

    for i_batch, batch_data in enumerate(train_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss = get_loss(model, batch_data)
        # nn_optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        # nn_optimizer.step()
        optimizer.step()

    for i_batch, batch_data in enumerate(test_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss, energy_loss, force_loss, stress_loss = \
            get_loss(model, batch_data, verbose=True)
        print(i, loss.cpu().detach().numpy(),
              energy_loss.cpu().detach().numpy(),
              force_loss.cpu().detach().numpy(),
              stress_loss.cpu().detach().numpy())
        if loss.cpu().detach().numpy() < min_loss:
            min_loss = loss.cpu().detach().numpy()
            torch.save(model.state_dict(), 'parameter-new.pkl')

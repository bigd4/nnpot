import logging
from ase.io import read
from ani.model import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData
from torch.utils.data import DataLoader
import torch
from ani.symmetry_functions import BehlerG1, CombinationRepresentation
from ani.cutoff import CosineCutoff
import torch.nn as nn


device = "cpu"

cutoff = 5.0
n_radius = 30
n_angular = 0
environment_provider = ASEEnvironment(cutoff)
cut_fn = CosineCutoff(cutoff)
rdf = BehlerG1(n_radius, cut_fn)

representation = CombinationRepresentation(rdf)
model = ANI(representation, environment_provider)
# model.load_state_dict(torch.load('parameter-new.pkl'))
frames = read('stress.traj', ':')
n_split = 120

train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=_collate_aseatoms)

loss_fn = torch.nn.MSELoss()
def get_loss(model, loss_fn, batch_data, weight=[1.0, 1.0, 1.0], verbose=False):
    w_energy, w_forces, w_stress = weight
    predict_energy = model.get_energies(batch_data)
    predict_forces = model.get_forces(batch_data)
    predict_stress = model.get_stresses(batch_data)
    target_energy = batch_data['energy']
    target_forces = batch_data['forces']
    target_stress = batch_data['stress']
    energy_loss = loss_fn(predict_energy, target_energy)
    force_loss = loss_fn(predict_forces, target_forces)
    stress_loss = loss_fn(predict_stress, target_stress)
    loss = w_energy * energy_loss + w_forces * force_loss + w_stress * stress_loss
    if verbose:
        return loss, energy_loss, force_loss, stress_loss
    return loss


nn_parameters, descriptor_parameters = [], []
for key, value in model.named_parameters():
    if 'etas' in key or 'rss' in key:
        descriptor_parameters.append(value)
    else:
        nn_parameters.append(value)

nn_optimizer = torch.optim.Adam(nn_parameters)
descriptor_optimizer = torch.optim.Adam(descriptor_parameters)

epoch = 1000
min_loss = 1000
for i in range(epoch):
    # optimize descriptor parameters
    if i % 5 == 0:
        for i_batch, batch_data in enumerate(train_loader):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            loss = get_loss(model, loss_fn, batch_data)
            descriptor_optimizer.zero_grad()
            loss.backward()
            descriptor_optimizer.step()

    for i_batch, batch_data in enumerate(train_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss = get_loss(model, loss_fn, batch_data)
        nn_optimizer.zero_grad()
        loss.backward()
        nn_optimizer.step()

    for i_batch, batch_data in enumerate(test_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss, energy_loss, force_loss, stress_loss = \
            get_loss(model, loss_fn, batch_data, verbose=True)
        print(i, loss.cpu().detach().numpy(),
              energy_loss.cpu().detach().numpy(),
              force_loss.cpu().detach().numpy(),
              stress_loss.cpu().detach().numpy())
        if loss.cpu().detach().numpy() < min_loss:
            min_loss = loss.cpu().detach().numpy()
            torch.save(model.state_dict(), 'parameter-new.pkl')

# for i_batch, batch_data in enumerate(test_loader):
#     batch_data = {k: v.to(device) for k, v in batch_data.items()}
# loss_fn = torch.nn.MSELoss()
#
# nn_parameters, descriptor_parameters = [], []
# for key, value in model.named_parameters():
#     if 'etas' in key or 'rss' in key:
#         descriptor_parameters.append(value)
#     else:
#         nn_parameters.append(value)
#
# nn_optimizer = torch.optim.Adam(nn_parameters)
# descriptor_optimizer = torch.optim.Adam(descriptor_parameters)
#

from ani.environment import ASEEnvironment
from ani.cutoff import CosineCutoff
from ani.jjnet import Representation, BehlerG1
from ase.io import read
from ani.dataloader import AtomsData
from ani.gpr import DataSet, GPR, RBF
from torch.utils.data import DataLoader
import torch
from ani.dataloader import get_dict, _collate_aseatoms


cutoff = 3.
a = read('dataset.traj',':')
environment_provider = ASEEnvironment(cutoff)
cut_fn = CosineCutoff(cutoff)
representation = Representation(30,0,cut_fn)

d = DataSet(environment_provider, representation)
d.update_dataset(a[:130])

kern = RBF()
model = GPR(representation, kern)
model.connect_dataset(d)
epoch = 100
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
test_data = AtomsData(a[130:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=_collate_aseatoms)

tmp_list = _collate_aseatoms([get_dict(atoms, environment_provider) for atoms in a[130:]])

nn_parameters, hyper_parameters = [], []
for key, value in model.named_parameters():
    if 'etas' in key or 'rss' in key:
        hyper_parameters.append(value)
    else:
        nn_parameters.append(value)

nn_optimizer = torch.optim.Adam(nn_parameters)
hyper_optimizer = torch.optim.Adam(hyper_parameters)

for i in range(10000):
    obj = model.compute_log_likelihood()
    nn_optimizer.zero_grad()
    obj.backward()
    nn_optimizer.step()
    if i % 50 == 0:
        obj = model.compute_log_likelihood()
        hyper_optimizer.zero_grad()
        obj.backward()
        hyper_optimizer.step()
    if i % 500 == 0:
        print(i, ':', obj.item())

batch_data = tmp_list

predict_energy, std = model(batch_data)
predict_energy = predict_energy.view(-1)
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


print(energy_loss, force_loss, stress_loss)

E1 = predict_energy.detach().numpy().reshape(-1)
E2 = target_energy.detach().numpy().reshape(-1)

F1 = predict_forces.detach().numpy().reshape(-1)
F2 = target_forces.detach().numpy().reshape(-1)

S1 = predict_stress.detach().numpy().reshape(-1)
S2 = target_stress.detach().numpy().reshape(-1)
import numpy as np
import matplotlib.pyplot as plt
mae_energies = np.mean(np.abs(E1 - E2))
r2_energies = 1 - np.sum((E1 - E2)**2) / \
    np.sum((E2 - np.mean(E2))**2)
mae_forces = np.mean(np.abs(F1 - F2))
r2_forces = 1 - np.sum((F1 - F2)**2) / \
    np.sum((F2 - np.mean(F2))**2)
mae_stress = np.mean(np.abs(S1 - S2))
r2_stress = 1 - np.sum((S1 - S2)**2) / \
    np.sum((S2 - np.mean(S2))**2)
import logging
from ase.io import read
from ani.jjnet import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData
from ani.train import Trainer
from torch.utils.data import DataLoader
import torch


logging.basicConfig(filename='log5.txt', level=logging.DEBUG,
                    format="%(asctime)s %(message)s", datefmt='%H:%M:%S')

# logging.info('cuda:{}\ndevice:{}'.format(torch.cuda.is_available(), torch.cuda.get_device_name(0)))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = "cpu"
cutoff = 5.0
n_radius = 30
n_angular = 0

net = ANI(n_radius, n_angular, cutoff)
from ani.jjnet import BehlerG1
from ani.cutoff import CosineCutoff
cut_fn = CosineCutoff(cutoff)
# etas = torch.load('parameter-new.pkl')['representation.functions.0.etas']
# rss = torch.load('parameter-new.pkl')['representation.functions.0.rss']
# net.representation.RDF = BehlerG1(30, cut_fn, etas, rss, False)
loss_calculator = torch.nn.MSELoss()
trainer = Trainer(net, loss_calculator)

frames = read('stress.traj', ':')
n_split = 120
environment_provider = ASEEnvironment(cutoff)
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=_collate_aseatoms)

trainer.train(1000, train_loader, test_loader, device)
# for i_batch, batch_data in enumerate(test_loader):
#     batch_data = {k: v.to(device) for k, v in batch_data.items()}
#
# nn_parameters, descriptor_parameters = [], []
# for key, value in net.named_parameters():
#     if 'etas' in key or 'rss' in key:
#         descriptor_parameters.append(value)
#     else:
#         nn_parameters.append(value)
#
# nn_optimizer = torch.optim.Adam(nn_parameters)
# descriptor_optimizer = torch.optim.Adam(descriptor_parameters)
#
# predict_energy = net(batch_data)
# predict_forces = -torch.autograd.grad(
#         predict_energy.sum(),
#         batch_data['positions'],
#         create_graph=True,
#         retain_graph=True
#     )[0]
# predict_stress = torch.autograd.grad(
#     predict_energy.sum(),
#     batch_data['scaling'],
#     create_graph=True,
#     retain_graph=True
# )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / batch_data['volume']
#
# nn_optimizer.zero_grad()
# descriptor_optimizer.zero_grad()
# target_energy = batch_data['energy']
# target_forces = batch_data['forces']
# target_stress = batch_data['stress']
# energy_loss = loss_calculator(predict_energy, target_energy)
# force_loss = loss_calculator(predict_forces, target_forces)
# stress_loss = loss_calculator(predict_stress, target_stress)
#
# loss = energy_loss + force_loss + stress_loss
# loss.backward()
# print(nn_parameters[0].grad)
# print(descriptor_parameters[0].grad)
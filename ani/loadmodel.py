from dataloader import AtomsData, _collate_aseatoms
from torch.utils.data import DataLoader
from ase.io import read
from environment import ASEEnvironment
from jjnet import ANI
import torch


cutoff = 3.0
w_energy = 1.0
w_forces = 1.0
epoch = 10000
frames = read('dataset.traj', ':')
environment_provider = ASEEnvironment(cutoff)
data = AtomsData(frames, environment_provider)
net = ANI()
net.load_state_dict(torch.load('model.pkl'))
loss_calculator = torch.nn.MSELoss()
data_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)

for i_batch, batch_data in enumerate(data_loader):
    predict_energy = net(batch_data)
    predict_forces = -torch.autograd.grad(predict_energy.sum(), batch_data['positions'], create_graph=True, retain_graph=True)[0]
    target_energy = batch_data['energy']
    target_forces = batch_data['forces']
    energy_loss = loss_calculator(predict_energy, target_energy)
    force_loss = loss_calculator(predict_forces, target_forces)
    loss = w_energy * energy_loss + w_forces * force_loss
    print(loss)

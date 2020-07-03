from dataloader import AtomsData, _collate_aseatoms
from torch.utils.data import DataLoader
from ase.io import read
from environment import ASEEnvironment
from jjnet import ANI
import torch


cutoff = 3.0
w_energy = 1.0
w_forces = 1.0
w_stress = 1.0
epoch = 10000
frames = read('dataset.traj', ':')
n_split = 130
environment_provider = ASEEnvironment(cutoff)
data = AtomsData(frames[:n_split], environment_provider)
net = ANI()
optimizer = torch.optim.Adam(net.parameters())
loss_calculator = torch.nn.MSELoss()
data_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)


###########################
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=_collate_aseatoms)
###########################
import numpy as np
l1,l2,l3,l4 = np.zeros([4,epoch])
m_loss = 100
for i in range(epoch):
    for i_batch, batch_data in enumerate(data_loader):
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
        )[0][:,[0,1,2,1,0,0],[0,1,2,2,2,1]]/batch_data['volume']

        target_energy = batch_data['energy']
        target_forces = batch_data['forces']
        target_stress = batch_data['stress']
        energy_loss = loss_calculator(predict_energy, target_energy)
        force_loss = loss_calculator(predict_forces, target_forces)
        stress_loss = loss_calculator(predict_stress, target_stress)

        loss = w_energy * energy_loss + w_forces * force_loss + w_stress * stress_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
        )[0][:,[0,1,2,1,0,0],[0,1,2,2,2,1]]/batch_data['volume']

        target_energy = batch_data['energy']
        target_forces = batch_data['forces']
        target_stress = batch_data['stress']
        energy_loss = loss_calculator(predict_energy, target_energy)
        force_loss = loss_calculator(predict_forces, target_forces)
        stress_loss = loss_calculator(predict_stress, target_stress)

        loss = w_energy * energy_loss + w_forces * force_loss + w_stress * stress_loss
        
        print('loss:',loss)
        print('energy_loss:',energy_loss)
        print('force_loss:',force_loss)
        print('stress_loss:',stress_loss)
        l1[i] = loss
        l2[i] = energy_loss
        l3[i] = force_loss
        l4[i] = stress_loss
        np.savez('loss.npz',l=l1,e=l2,f=l3,s=l4)
        if loss < m_loss:
            m_loss = loss
            torch.save(net.state_dict(), 'parameter.pkl')


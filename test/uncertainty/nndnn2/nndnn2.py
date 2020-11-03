from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, _collate_aseatoms
from ani.model import ANI, DeltaANI
from ani.symmetry_functions import *
from ani.utils import *
from torch.utils.data import DataLoader
import torch
import logging
from ani.kalmanfilter import KalmanFilter
import time
import matplotlib.pyplot as plt


logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(asctime)s  %(message)s",datefmt='%H:%M:%S')
device = "cpu"

# read data set and get elements
np.random.seed(1)
frames = read('../sps_all.xyz', ':')
np.random.shuffle(frames)

elements = get_elements(frames)
mean, std = get_statistic(frames)

# set cutoff and environment_provider
cutoff = 3.
environment_provider = ASEEnvironment(cutoff)

n_split1 = 900
n_split2 = 1000
train_data = AtomsData(frames[:n_split1], environment_provider)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=_collate_aseatoms)
delta_train_data = AtomsData(frames[n_split1: n_split2], environment_provider)
delta_train_loader = DataLoader(delta_train_data, batch_size=128, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split2: 1500], environment_provider)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=_collate_aseatoms)

# model 1
n_radius = 22
n_angular = 5
cut_fn = CosineCutoff(cutoff)
rss = torch.linspace(0.3, cutoff-0.3, n_radius)
etas = 0.5 * torch.ones_like(rss) / (rss[1] - rss[0]) ** 2
rdf = BehlerG1(elements, n_radius, cut_fn, etas=etas, rss=rss, train_para=False)

etas = 0.5 / torch.linspace(1, cutoff-0.3, n_angular) ** 2
adf = BehlerG3(elements, n_angular, cut_fn, etas=etas, train_para=False)

representation = CombinationRepresentation(rdf, adf)

t0 = time.time()
for atoms_data in train_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    atoms_data['representations'] = representation(batch_data).squeeze(0)

for atoms_data in delta_train_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    atoms_data['representations'] = representation(batch_data).squeeze(0)

for atoms_data in test_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    atoms_data['representations'] = representation(batch_data).squeeze(0)

model = ANI(representation, elements, [15, 15], mean=mean, std=std)
h = lambda batch_data: model.get_energies(batch_data) / batch_data['n_atoms']
z = lambda batch_data: batch_data['energy'] / batch_data['n_atoms']
optimizer = KalmanFilter(model.parameters(), h, z, eta_0=1e-3, eta_tau=2.)
epoch = 10
min_loss = 1000
eloss = []
for i in range(epoch):
    optimizer.step(train_loader)
    with torch.no_grad():
        loss = 0.
        for i_batch, batch_data in enumerate(test_loader):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            loss += get_loss(model, batch_data, weight=[1.0, 0.0, 0.0]).item()
        loss = np.sqrt(loss / (i_batch + 1))
    eloss.append(loss)
    print('{:5d}\t{:.4f}'.format(i, loss))
    if loss < min_loss:
        min_loss = loss
        torch.save(model.state_dict(), 'para.pt')
        torch.save(model, 'model.pkl')

# model 2
delta_model = DeltaANI(representation, elements, [15, 15])
optimizer = torch.optim.Adam(delta_model.parameters())
epoch = 200
min_loss = 100000
for i in range(epoch):
    for i_batch, batch_data in enumerate(delta_train_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        predict_var = delta_model.get_variance(batch_data) / batch_data['n_atoms']
        target_var = (batch_data['energy'] - model.get_energies(batch_data)) ** 2 / batch_data['n_atoms']
        loss = torch.sum(torch.log(predict_var) + target_var / (predict_var + 1e-8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        var_loss = 0.
        for i_batch, batch_data in enumerate(test_loader):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            predict_var = delta_model.get_variance(batch_data) / batch_data['n_atoms']
            target_var = (batch_data['energy'] - model.get_energies(batch_data)) ** 2 / batch_data['n_atoms']
            var_loss += torch.sum(torch.log(predict_var) + target_var / (predict_var + 1e-8))

    print('{:5d}\t{:.4f}'.format(i, var_loss))

    if var_loss < min_loss:
        min_loss = var_loss
        torch.save(delta_model.state_dict(), 'delta_para.pt')
        torch.save(delta_model, 'delta_model.pkl')

print(time.time() - t0)
predict_var, target_var, predict_e, target_e, n_atoms = [], [], [], [], []
for atoms_data in test_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    n = batch_data['n_atoms'].item()
    n_atoms.append(n)
    predict_e.append(model.get_energies(batch_data).item())
    target_e.append(batch_data['energy'].item())
    predict_var.append(delta_model.get_variance(batch_data).item() / n)
    target_var.append((model.get_energies(batch_data).item() - batch_data['energy'].item()) ** 2 / n)
predict_var = np.array(predict_var)
target_var = np.array(target_var)
predict_e = np.array(predict_e)
target_e = np.array(target_e)
n_atoms = np.array(n_atoms)

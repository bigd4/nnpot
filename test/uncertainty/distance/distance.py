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
from scipy.spatial.distance import cdist
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


model = torch.load('model.pkl')

X_train = []
Y_train = []
n_train = []
# model 2
for atoms_data in train_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    lv = model.get_latent_variables(batch_data).squeeze(0).detach().numpy()
    residual = (model.get_energies(batch_data) - batch_data['energy']).detach().squeeze(0).numpy()
    n_train.append(batch_data['n_atoms'].item())
    X_train.append(lv)
    Y_train.append(residual)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
n_train = np.array(n_train)

X_delta = []
Y_delta = []
n_delta = []
# model 2
for atoms_data in delta_train_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    lv = model.get_latent_variables(batch_data).squeeze(0).detach().numpy()
    residual = (model.get_energies(batch_data) - batch_data['energy']).detach().squeeze(0).numpy()
    n_delta.append(batch_data['n_atoms'].item())
    X_delta.append(lv)
    Y_delta.append(residual)
X_delta = np.array(X_delta)
Y_delta = np.array(Y_delta)
n_delta = np.array(n_delta)

dis = cdist(X_delta/n_delta.reshape(-1, 1), X_train/n_train.reshape(-1, 1))
dis = np.mean(np.sort(dis, axis=1)[:, :10], axis=1)



# X_test = []
# Y_test = []
# n_test = []
# for atoms_data in test_data:
#     batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
#     lv = model.get_latent_variables(batch_data).squeeze(0).detach().numpy()
#     residual = (model.get_energies(batch_data) - batch_data['energy']).detach().squeeze(0).numpy()
#     n_test.append(batch_data['n_atoms'].item())
#     X_test.append(lv)
#     Y_test.append(residual)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)
# n_test = np.array(n_test)
# print(time.time() - t0)

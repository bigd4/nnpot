from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from ani.model import DropoutANI
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
from torch.utils.data import DataLoader
import torch
import logging


logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(asctime)s  %(message)s",datefmt='%H:%M:%S')
device = "cpu"
# device = 'cuda:0'
# read data set and get elements
np.random.seed(1)
frames = read('../sps_all.xyz', ':')
np.random.shuffle(frames)

elements = get_elements(frames)
mean, std = get_statistic(frames)

# set cutoff and environment_provider
cutoff = 3.5
environment_provider = ASEEnvironment(cutoff)

# behler
n_radius = 30
n_angular = 10
cut_fn = CosineCutoff(cutoff)
rss = torch.linspace(0.3, cutoff-0.3, n_radius)
etas = 0.5 * torch.ones_like(rss) / (rss[1] - rss[0]) ** 2
rdf = BehlerG1(elements, n_radius, cut_fn, etas=etas, rss=rss, train_para=False)

etas = 0.5 / torch.linspace(1, cutoff-0.3, n_angular) ** 2
adf = BehlerG3(elements, n_angular, cut_fn, etas=etas, train_para=False)

# get representations
representation = CombinationRepresentation(rdf, adf)

model = DropoutANI(representation, elements, [70, 70], mean=mean, std=std, p=0.3)
model.to(device)

n_split = 1000
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=_collate_aseatoms)

for atoms_data in train_data:
    batch_data = {k: v.unsqueeze(0).to(device) for k, v in atoms_data.items()}
    atoms_data['representations'] = representation(batch_data).squeeze(0)

for atoms_data in test_data:
    batch_data = {k: v.unsqueeze(0).to(device) for k, v in atoms_data.items()}
    atoms_data['representations'] = representation(batch_data).squeeze(0)

optimizer = torch.optim.Adam(model.parameters())

epoch = 500
min_loss = 1000
eloss = []
for i in range(epoch):
    for i_batch, batch_data in enumerate(train_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss = get_loss(model, batch_data, weight=[1.0, 0.0, 0.0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

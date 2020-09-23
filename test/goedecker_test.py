from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from ani.model import ANI, GoedeckerNet
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
from torch.utils.data import DataLoader
import torch

torch.cuda.empty_cache()
device = "cpu"
# device = 'cuda:0'
# read data set and get elements
frames = read('Al/base.traj', ':')
# frames = augment_data(frames, n=300)
elements = []
for atoms in frames:
    atoms.info['q_tot'] = 0.
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))

# set cutoff and environment_provider
cutoff = 3.
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

model = GoedeckerNet(representation, elements)
model.to(device)

n_split = int(0.8 * len(frames))
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=_collate_aseatoms)

optimizer = torch.optim.Adam(model.parameters())
batch_data = convert_frames(frames[:8], environment_provider)
epoch = 1000
min_loss = 1000
# loss = get_loss(model, batch_data, weight=[1.0, 0.0, 0.0])
# optimizer.zero_grad()
# loss.backward()
# loss = get_loss(model, batch_data, weight=[1.0, 0.0, 0.0])
for i in range(epoch):
    # optimize descriptor parameters
    for i_batch, batch_data in enumerate(train_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss = get_loss(model, batch_data, weight=[1.0, 0.0, 0.0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss, energy_loss, force_loss, stress_loss = 0., 0., 0., 0.
    for i_batch, batch_data in enumerate(test_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss, energy_loss, force_loss, stress_loss = \
            get_loss(model, batch_data, weight=[1.0, 0.0, 0.0], verbose=True)

    print(i, loss.cpu().detach().numpy(),
          energy_loss.cpu().detach().numpy(),
          force_loss.cpu().detach().numpy(),
          stress_loss.cpu().detach().numpy())
    if loss.cpu().detach().numpy() < min_loss:
        min_loss = loss.cpu().detach().numpy()
        torch.save(model.state_dict(), 'parameter-new.pkl')

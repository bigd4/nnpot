from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames
from ani.model import ANI
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
from torch.utils.data import DataLoader
import torch

device = "cpu"
# read data set and get elements
frames = read('../test/initpop1.traj', ':')
# frames = augment_data(frames, n=300)
elements = []
for atoms in frames:
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))

# set cutoff and environment_provider
cutoff = 5.
environment_provider = ASEEnvironment(cutoff)

# behler
n_radius = 30
n_angular = 10
cut_fn = CosineCutoff(cutoff)
rdf = BehlerG1(n_radius, cut_fn)
adf = BehlerG3(n_angular, cut_fn)
# zernike
zer = Zernike(elements, 8, 8, False, cutoff)

# dpmd
n_radius = 30
n_angular = 10
cutoff_smooth = 4.8
dpmd_r = Deepmd_radius(n_radius, SmoothCosineCutoff(cutoff_smooth, cutoff))
dpmd_a = Deepmd_angular(n_angular, SmoothCosineCutoff(cutoff_smooth, cutoff))

# get representations
representation = CombinationRepresentation(rdf)

model = ANI(representation, environment_provider)

n_split = 120
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
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
        loss = get_loss(model, batch_data)
        nn_optimizer.zero_grad()
        loss.backward()
        nn_optimizer.step()

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

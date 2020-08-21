from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames
from ani.model import *
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
import torch
import torch.nn as nn


# read data set and get elements
frames = read('../test/dataset.traj', ':')
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
representation = CombinationRepresentation(zer)
# representation = CombinationRepresentation(rdf)

# set kernel and prior
kern = RBF(dimension=representation.dimension)
prior = RepulsivePrior(r_max=cutoff)

# get model
model = GPR(representation, kern, environment_provider)
# model = ANI(representation, environment_provider, train_descriptor=True)

n_split = int(len(frames) * 0.8)
model.update_dataset(frames[:n_split])
model.train(3000)
torch.save(model.state_dict(), 'parameter.pkl')

loss_fn = torch.nn.MSELoss()
batch_data = convert_frames(frames[-8:], environment_provider)
batch_data['positions'].requires_grad_()
batch_data['scaling'].requires_grad_()
optimizer = torch.optim.Adam([batch_data['positions'], batch_data['scaling']])

epoch = 1000
min_loss = 1000
for i in range(10000):
    energies = torch.sum(model.get_energies(batch_data))
    optimizer.zero_grad()
    energies.backward()
    optimizer.step()
    forces = model.get_forces(batch_data).detach()
    forces = forces.view(forces.size()[0], -1).detach()
    print(torch.max(forces, 1)[0])


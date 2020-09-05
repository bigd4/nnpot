from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from torch.utils.data import DataLoader
from ani.model import GPR
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
import torch

# read data set and get elements
frames = read('../test/stress.traj', ':')
frames = augment_data(frames, n=300)
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

data = AtomsData(frames, environment_provider)
data_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
for i_batch, batch_data in enumerate(data_loader):
    print(i_batch)
    descriptor = representation(batch_data).sum(1).detach()
    energies = batch_data['energy'].unsqueeze(-1)

    if i_batch > 0:
        X_array = torch.cat((X_array, descriptor))
        y = torch.cat((y, energies))
    else:
        X_array = descriptor
        y = energies

import numpy as np
np.savez('des.npz',X_array=X_array,y=y)

import logging
from ase.io import read
from ani.model import ANI, GPR
from ani.kernel import RBF
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData, convert_frames
from torch.utils.data import DataLoader
import torch
from ani.symmetry_functions import BehlerG1, CombinationRepresentation
from ani.cutoff import CosineCutoff
import torch.nn as nn


device = "cpu"

cutoff = 5.0
n_radius = 30
n_angular = 0

cut_fn = CosineCutoff(cutoff)
environment_provider = ASEEnvironment(cutoff)
rdf = BehlerG1(n_radius, cut_fn)

representation = CombinationRepresentation(rdf)
model = ANI(representation, environment_provider)
frames = read('initpop1.traj', ':')
n_split = 120

kern = RBF(dimension=30)
gprmodel = GPR(representation, kern, environment_provider)
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=_collate_aseatoms)

model.update_dataset(frames)
model.train(1000)
gprmodel.update_dataset(frames)
gprmodel.train(30000)

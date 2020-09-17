import logging
from ase.io import read
from ani.model import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData, convert_frames
from torch.utils.data import DataLoader
import torch
from ani.symmetry_functions import BehlerG1, BehlerG3, CombinationRepresentation
from ani.cutoff import CosineCutoff
from ani.utils import get_loss
import torch.nn as nn
import numpy as np


device = "cpu"

frames = read('small_qm9.traj', ':')
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

representation = CombinationRepresentation(rdf, adf)
model = ANI(representation, elements, [50, 50])
batch_data = convert_frames(frames[:16], environment_provider)
f = model.get_energies(batch_data)

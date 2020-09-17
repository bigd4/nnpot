from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from ani.model import ANI
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
frames = read('initpop1.traj', ':')
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

# get representations
representation = CombinationRepresentation(rdf, adf)

model = ANI(representation, elements, [50, 50])
model.to(device)

n_split = int(0.8 * len(frames))
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=_collate_aseatoms)

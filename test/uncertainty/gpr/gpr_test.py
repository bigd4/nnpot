from ani.environment import ASEEnvironment
from ani.cutoff import CosineCutoff
from ase.io import read
from ani.dataloader import AtomsData, convert_frames
from ani.model import GPR
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from torch.utils.data import DataLoader
import torch
from ani.dataloader import get_dict, _collate_aseatoms

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
delta_train_data = AtomsData(frames[: n_split2], environment_provider)
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

kern = RBF(dimension=representation.dimension)
prior = RepulsivePrior(r_max=cutoff)
model = GPR(representation, kern, environment_provider)

model.data.update_data(frames[:n_split2])
model.train(100)
torch.save(model.state_dict(), 'parameter-gpr.pkl')

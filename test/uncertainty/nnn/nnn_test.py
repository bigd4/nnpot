from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from ani.model import ANI, NNEnsemble
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.kalmanfilter import KalmanFilter
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
cutoff = 3.
environment_provider = ASEEnvironment(cutoff)

# behler
n_radius = 22
n_angular = 5
cut_fn = CosineCutoff(cutoff)
rss = torch.linspace(0.3, cutoff-0.3, n_radius)
etas = 0.5 * torch.ones_like(rss) / (rss[1] - rss[0]) ** 2
rdf = BehlerG1(elements, n_radius, cut_fn, etas=etas, rss=rss, train_para=False)

etas = 0.5 / torch.linspace(1, cutoff-0.3, n_angular) ** 2
adf = BehlerG3(elements, n_angular, cut_fn, etas=etas, train_para=False)

# get representations
representation = CombinationRepresentation(rdf, adf)

n_bagging = 10
nets, optimizers = [], []
for _ in range(n_bagging):
    model = ANI(representation, elements, [15, 15])
    h = lambda batch_data, model=model: model.get_energies(batch_data) / batch_data['n_atoms']
    z = lambda batch_data: batch_data['energy'] / batch_data['n_atoms']
    optimizer = KalmanFilter(model.parameters(), h, z, eta_0=1e-3, eta_tau=2.3, q_tau=2.3)
    nets.append(model)
    optimizers.append(optimizer)

model = NNEnsemble(nets)

n_split = 1000
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=_collate_aseatoms)

for atoms_data in train_data:
    batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
    atoms_data['representations'] = representation(batch_data).squeeze(0)

epoch = 10
for optimizer in optimizers:
    for i in range(epoch):
        print(i)
        optimizer.step(train_loader)


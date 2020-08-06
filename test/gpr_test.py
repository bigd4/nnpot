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


frames = read('stress.traj', ':')
elements = []
for atoms in frames:
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))

cutoff = 5.
environment_provider = ASEEnvironment(cutoff)
cut_fn = CosineCutoff(cutoff)

rdf = BehlerG1(30, cut_fn)
zer = Zernike(elements, 8, 8, False, cutoff)
representation = CombinationRepresentation(zer)
kern = RBF(dimension=representation.dimension)
prior = RepulsivePrior(r_max=cutoff)
model = GPR(representation, kern, environment_provider, prior)

model.update_dataset(frames[:120])
model.train(50000)
torch.save(model.state_dict(), 'parameter-gpr.pkl')

loss_fn = torch.nn.MSELoss()
batch_data = convert_frames(frames[120:128], environment_provider)


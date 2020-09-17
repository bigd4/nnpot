from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames
from ani.model import SparseGPR, GPR
from ani.kernel import *
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
import torch

# read data set and get elements
frames = read('dataset.traj', ':')
# frames = augment_data(frames, n=300)
elements = []
for atoms in frames:
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))

# set cutoff and environment_provider
cutoff = 5.
environment_provider = ASEEnvironment(cutoff)

# zernike
zer = Zernike(elements, 8, 8, False, cutoff)

# get representations
representation = CombinationRepresentation(zer)

# set kernel and prior
# kern = RBF(dimension=representation.dimension)
kern = Hehe()
#prior = RepulsivePrior(r_max=cutoff)
prior = None

# get model
model1 = GPR(representation, kern, environment_provider, prior)
n_split = int(len(frames) * 0.8)
n_split = 120
model1.update_data(frames[:n_split])
model1.train(1000, 50)
batch_data = convert_frames(frames[-4:], environment_provider)
print(model1.get_energies(batch_data))

n_sparse = 50
model2 = SparseGPR(representation, kern, environment_provider, prior, n_sparse)
n_split = int(len(frames) * 0.8)
n_split = 120
model2.update_data(frames[:n_split])
model2.train(500, 50)
batch_data = convert_frames(frames[-4:], environment_provider)
print(model2.get_energies(batch_data))

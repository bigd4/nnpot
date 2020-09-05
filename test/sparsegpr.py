from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames
from ani.model import SparseGPR
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
import torch

# read data set and get elements
frames = read('stress.traj', ':')
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
kern = RBF(dimension=representation.dimension)
#prior = RepulsivePrior(r_max=cutoff)
prior = None

# get model
n_sparse = 200
model = SparseGPR(representation, kern, environment_provider, prior, n_sparse)

n_split = int(len(frames) * 0.8)
n_split = 40
model.update_data(frames[:n_split])
model.train(8000, 500)
batch_data = convert_frames(frames[:8], environment_provider)
print(model.get_energies(batch_data))

from ani.environment import ASEEnvironment
from ani.cutoff import CosineCutoff
from ase.io import read
from ani.dataloader import AtomsData, convert_frames
from ani.model import GPR
from ani.kernel import RBF
from ani.symmetry_functions import BehlerG1, CombinationRepresentation
from torch.utils.data import DataLoader
import torch
from ani.dataloader import get_dict, _collate_aseatoms


cutoff = 5.
environment_provider = ASEEnvironment(cutoff)
cut_fn = CosineCutoff(cutoff)


etas = torch.load('parameter-new.pkl')['representation.functions.0.etas']
rss = torch.load('parameter-new.pkl')['representation.functions.0.rss']
rdf = BehlerG1(30, cut_fn, etas, rss, False)
representation = CombinationRepresentation(rdf)
kern = RBF(dimension=30)
model = GPR(representation, kern, environment_provider)

a = read('stress.traj', ':')
model.update_dataset(a[:120])
model.train(30000)


loss_fn = torch.nn.MSELoss()
batch_data = convert_frames(a[120:], environment_provider)

predict_energy = model.get_energies(batch_data)
predict_forces = model.get_forces(batch_data)
predict_stress = model.get_stresses(batch_data)
target_energy = batch_data['energy']
target_forces = batch_data['forces']
target_stress = batch_data['stress']
energy_loss = loss_fn(predict_energy, target_energy)
force_loss = loss_fn(predict_forces, target_forces)
stress_loss = loss_fn(predict_stress, target_stress)

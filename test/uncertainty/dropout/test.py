from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from ani.model import DropoutANI
from ani.kernel import RBF
from ani.symmetry_functions import *
from ani.prior import RepulsivePrior
from ani.utils import *
from torch.utils.data import DataLoader
import torch
import logging
import numpy as np
from scipy import stats
from tqdm import tqdm


logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(asctime)s  %(message)s",datefmt='%H:%M:%S')
device = "cpu"
device = 'cuda:0'
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

model = DropoutANI(representation, elements, [50, 50], mean=mean, std=std, p=0.3)
model.to(device)

n_split = 4500
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=_collate_aseatoms)

model.load_state_dict(torch.load('para.pt'))

target_energies = []
predict_energies = []
predict_deltas = []
target_deltas = []
with torch.no_grad():
    for atoms_data in test_data:
        batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
        predict_energies.append(model.get_energies(batch_data).item() / batch_data['n_atoms'].item())
        target_energies.append(batch_data['energy'].item() / batch_data['n_atoms'].item())
        predict_deltas.append(delta_model.get_energies(batch_data).item() / batch_data['n_atoms'].item())
        target_deltas.append(predict_energies[-1] - target_energies[-1])

target_energies = np.array(target_energies)
predict_energies = np.array(predict_energies)
predict_deltas = np.array(predict_deltas)
target_deltas = np.array(target_deltas)

# Define a normalized bell curve we'll be using to calculate calibration
norm = stats.norm(loc=0, scale=1)

# Computing calibration
def calculate_density(percentile):
    # Find the normalized bounds of this percentile
    upper_bound = norm.ppf(percentile)

    # Normalize the residuals so they all should fall on the normal bell curve
    normalized_residuals = target_deltas.reshape(-1) / predict_deltas.reshape(-1)

    # Count how many residuals fall inside here
    num_within_quantile = 0
    for resid in normalized_residuals:
        if resid <= upper_bound:
            num_within_quantile += 1

    # Return the fraction of residuals that fall within the bounds
    density = num_within_quantile / len(target_deltas)
    return density

predicted_pi = np.linspace(0, 1, 100)
observed_pi = [calculate_density(quantile)
               for quantile in tqdm(predicted_pi, desc='Calibration')]

calibration_error = ((predicted_pi - observed_pi)**2).sum()
print('Calibration error = %.2f' % calibration_error)

import matplotlib.pyplot as plt
plt.scatter(predicted_pi, predicted_pi)
plt.scatter(predicted_pi, observed_pi)

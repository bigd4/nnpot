import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

des = np.load('des.npz')
# Training data is 100 points in [0,1] inclusive regularly spaced
data_x = torch.tensor(des['X_array'])
mean = torch.mean(data_x, 0)
std = torch.mean(data_x, 0)
data_x = (data_x - mean) / std
# True function is sin(2*pi*x) with Gaussian noise
data_y = torch.tensor(des['y']).view(-1)

n_split = int(0.8 * data_x.size()[0])
train_x = data_x[:n_split]
train_y = data_y[:n_split]


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

training_iter = 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()



from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData, convert_frames, _collate_aseatoms
from torch.utils.data import DataLoader
from ani.symmetry_functions import *
from ani.utils import *
import torch

# read data set and get elements
frames = read('../test/stress.traj', ':')
frames = augment_data(frames, n=300)
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
representation.mean = mean
representation.std = std

data = AtomsData(frames[100:128], environment_provider)
data_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
for i_batch, batch_data in enumerate(data_loader):
    batch_data['positions'].requires_grad_()
    descriptor = representation(batch_data).sum(1)
    energies = model(descriptor).mean
    forces = -torch.autograd.grad(
        energies.sum(),
        batch_data['positions']
    )[0]




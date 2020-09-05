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

n_split = int(0.8*data_x.size()[0])
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
        zz = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        print(zz.mean.size())
        return zz

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

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = data_x[n_split:]
    test_y = data_y[n_split:]
    f_preds = model(test_x)
    f_mean = f_preds.mean
    f_var = f_preds.variance

plt.scatter(test_y.numpy(), f_mean.numpy())
plt.errorbar(test_y.numpy(), f_mean.numpy(), yerr=np.sqrt(f_var.numpy()), fmt="o")

test_x.requires_grad_()
mean_x = model.mean_module(test_x)
covar_x = model.covar_module(test_x)
zz = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
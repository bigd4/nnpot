from ani.environment import ASEEnvironment
from ani.cutoff import *
from ase.io import read
from ani.dataloader import AtomsData
import torch
import logging
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor


def calculate_density(percentile, target_deltas, predict_deltas):
    # Find the normalized bounds of this percentile
    norm = stats.norm(loc=0, scale=1)
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


def get_calibration(target_deltas, predict_deltas):
    predicted_pi = np.linspace(0, 1, 100)
    observed_pi = [calculate_density(quantile, target_deltas, predict_deltas)
                   for quantile in tqdm(predicted_pi, desc='Calibration')]
    return predicted_pi, observed_pi


def get_array(test_data, f1, f2, f3, f4):
    target_energies = []
    predict_energies = []
    predict_deltas = []
    target_deltas = []
    with torch.no_grad():
        for atoms_data in test_data:
            batch_data = {k: v.unsqueeze(0) for k, v in atoms_data.items()}
            predict_energies.append(f1(batch_data))
            target_energies.append(f2(batch_data))
            predict_deltas.append(f3(batch_data))
            target_deltas.append(f4(batch_data))

    target_energies = np.array(target_energies)
    predict_energies = np.array(predict_energies)
    predict_deltas = np.array(predict_deltas)
    target_deltas = np.array(target_deltas)
    return target_energies, predict_energies, predict_deltas, target_deltas


def nndnn(test_data):
    model = torch.load('nndnn/model.pkl')
    delta_model = torch.load('nndnn/delta_model.pkl')

    f1 = lambda batch_data: model.get_energies(batch_data).item() / batch_data['n_atoms'].item()
    f2 = lambda batch_data: batch_data['energy'].item() / batch_data['n_atoms'].item()
    f3 = lambda batch_data: abs(delta_model.get_energies(batch_data).item()) / batch_data['n_atoms'].item()
    f4 = lambda batch_data: (f1(batch_data) - f2(batch_data))

    return get_array(test_data, f1, f2, f3, f4)


def nndnn2(test_data):
    model = torch.load('nndnn2/model.pkl')
    delta_model = torch.load('nndnn2/delta_model.pkl')

    f1 = lambda batch_data: model.get_energies(batch_data).item() / batch_data['n_atoms'].item()
    f2 = lambda batch_data: batch_data['energy'].item() / batch_data['n_atoms'].item()
    f3 = lambda batch_data: np.sqrt(delta_model.get_variance(batch_data).item() / batch_data['n_atoms'].item())
    f4 = lambda batch_data: (f1(batch_data) - f2(batch_data)) * np.sqrt(batch_data['n_atoms'].item())

    return get_array(test_data, f1, f2, f3, f4)


def dropout(test_data):
    model = torch.load('dropout/model.pkl')
    for atoms_data in test_data:
        batch_data = {k: v.unsqueeze(0).to(device) for k, v in atoms_data.items()}
        atoms_data['representations'] = model.representation(batch_data).squeeze(0)

    f1 = lambda batch_data: model.get_energies(batch_data).item() / batch_data['n_atoms'].item()
    f2 = lambda batch_data: batch_data['energy'].item() / batch_data['n_atoms'].item()
    f3 = lambda batch_data: model.get_energies(batch_data, True)[1].item() / np.sqrt(batch_data['n_atoms'].item())
    f4 = lambda batch_data: (f1(batch_data) - f2(batch_data)) * np.sqrt(batch_data['n_atoms'].item())

    return get_array(test_data, f1, f2, f3, f4)


def bayeslast(test_data):
    model = torch.load('bayeslast/model.pkl')
    X_train = np.load('bayeslast/X_train.npy')
    Y_train = np.load('bayeslast/Y_train.npy')
    model2 = BayesianRidge()
    model2.fit(X_train, Y_train)

    f1 = lambda batch_data: model.get_energies(batch_data).item() / batch_data['n_atoms'].item()
    f2 = lambda batch_data: batch_data['energy'].item() / batch_data['n_atoms'].item()
    f3 = lambda batch_data: model2.predict(model.get_latent_variables(batch_data).detach().numpy(), True)[1][0] \
                            / np.sqrt(batch_data['n_atoms'].item())
    f4 = lambda batch_data: (f1(batch_data) - f2(batch_data)) * np.sqrt(batch_data['n_atoms'].item())

    return get_array(test_data, f1, f2, f3, f4)


def GPRlast(test_data):
    model = torch.load('bayeslast/model.pkl')
    X_train = np.load('bayeslast/X_train.npy')
    Y_train = np.load('bayeslast/Y_train.npy')
    model2 = GaussianProcessRegressor()
    model2.fit(X_train, Y_train)

    f1 = lambda batch_data: model.get_energies(batch_data).item() / batch_data['n_atoms'].item()
    f2 = lambda batch_data: batch_data['energy'].item() / batch_data['n_atoms'].item()
    f3 = lambda batch_data: model2.predict(model.get_latent_variables(batch_data).detach().numpy(), True)[1][0] \
                            / np.sqrt(batch_data['n_atoms'].item())
    f4 = lambda batch_data: (f1(batch_data) - f2(batch_data)) * np.sqrt(batch_data['n_atoms'].item())

    return get_array(test_data, f1, f2, f3, f4)


def GPRpriorlast(test_data):
    model = torch.load('bayeslast/model.pkl')
    X_train = np.load('bayeslast/X_train.npy')
    Y_train = np.load('bayeslast/Y_train.npy')
    Y_mean = np.load('bayeslast/Y_mean.npy')
    model2 = GaussianProcessRegressor()
    model2.fit(X_train, Y_train - Y_mean)

    def f1(batch_data):
        prior = model.get_energies(batch_data).item()
        representations = model.get_latent_variables(batch_data).detach().numpy()
        attach = model2.predict(representations)[0]
        energy = (prior + attach) / batch_data['n_atoms'].item()
        return energy

    f2 = lambda batch_data: batch_data['energy'].item() / batch_data['n_atoms'].item()
    f3 = lambda batch_data: model2.predict(model.get_latent_variables(batch_data).detach().numpy(), True)[1][0] \
                            / np.sqrt(batch_data['n_atoms'].item())
    f4 = lambda batch_data: (f1(batch_data) - f2(batch_data)) * np.sqrt(batch_data['n_atoms'].item())

    return get_array(test_data, f1, f2, f3, f4)

logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(asctime)s  %(message)s",datefmt='%H:%M:%S')
device = "cpu"

# read data set and get elements
np.random.seed(1)
frames = read('sps_all.xyz', ':')
np.random.shuffle(frames)
frames = frames[1000:]
cutoff = 3.
environment_provider = ASEEnvironment(cutoff)
test_data = AtomsData(frames, environment_provider)

target_energies, predict_energies, predict_deltas, target_deltas = nndnn2(test_data)
predicted_pi, observed_pi = get_calibration(target_deltas, predict_deltas)
plt.scatter(predict_deltas, target_deltas)
plt.figure()
plt.scatter(predicted_pi, observed_pi)

# target_energies, predict_energies, predict_deltas, target_deltas = GPRlast(test_data)
# predicted_pi, observed_pi = get_calibration(target_deltas, predict_deltas)
# plt.scatter(predict_deltas, target_deltas)
# plt.figure()
# plt.scatter(predicted_pi, observed_pi)

# plt.figure()
# plt.subplot(2, 2, 1)
# plt.title('nndnn')
# target_energies, predict_energies, predicted_pi, observed_pi = nndnn(test_data)
# calibration_error = ((predicted_pi - observed_pi)**2).sum()
# plt.scatter(predicted_pi, observed_pi, label=str(calibration_error))
# plt.scatter(predicted_pi, predicted_pi)
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.title('dropout')
# target_energies, predict_energies, predicted_pi, observed_pi = dropout(test_data)
# calibration_error = ((predicted_pi - observed_pi)**2).sum()
# plt.scatter(predicted_pi, observed_pi, label=str(calibration_error))
# plt.scatter(predicted_pi, predicted_pi)
# plt.legend()
#
# # plt.subplot(2, 2, 3)
# # plt.title('bayeslast')
# # target_energies, predict_energies, predicted_pi, observed_pi = bayeslast(test_data)
# # calibration_error = ((predicted_pi - observed_pi)**2).sum()
# # plt.scatter(predicted_pi, observed_pi, label=str(calibration_error))
# # plt.scatter(predicted_pi, predicted_pi)
# # plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.title('GPRlast')
# target_energies, predict_energies, predicted_pi, observed_pi = GPRlast(test_data)
# calibration_error = ((predicted_pi - observed_pi)**2).sum()
# plt.scatter(predicted_pi, observed_pi, label=str(calibration_error))
# plt.scatter(predicted_pi, predicted_pi)
# plt.legend()
#
#
# plt.subplot(2, 2, 3)
# plt.title('GPRpriorlast')
# target_energies, predict_energies, predicted_pi, observed_pi = GPRpriorlast(test_data)
# calibration_error = ((predicted_pi - observed_pi)**2).sum()
# plt.scatter(predicted_pi, observed_pi, label=str(calibration_error))
# plt.scatter(predicted_pi, predicted_pi)
# plt.legend()

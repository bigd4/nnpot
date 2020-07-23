from ani.environment import ASEEnvironment
from ani.kernel import RBF
from ani.cutoff import CosineCutoff
from ani.model import ANI, GPR
from ani.dataloader import convert_frames
from ani.symmetry_functions import BehlerG1, CombinationRepresentation
from torch.utils.data import DataLoader
import torch
import numpy as np
from ase.io import read


class pytorchGPRmodel:
    def __init__(self):
        cutoff = 5.0
        n_radius = 30
        self.environment_provider = ASEEnvironment(cutoff)
        cut_fn = CosineCutoff(cutoff)

        rdf = BehlerG1(n_radius, cut_fn)
        representation = CombinationRepresentation(rdf)
        self.ani_model = ANI(representation, self.environment_provider)
        kern = RBF()
        self.model = GPR(representation, kern, self.environment_provider)

    def train(self, epoch1=10, epoch2=30000):
        self.ani_model.load_state_dict(torch.load('tmp.pkl'))
        self.model.recompute_X_array()
        self.model.train(epoch2)
        tmp = self.model.kern.variance.detach().numpy()
        self.K0 = np.log(1 + np.exp(tmp))

    def get_loss(self, images):
        batch_data = convert_frames(images, self.environment_provider)
        predict_energy = self.model.get_energies(batch_data).detach().numpy()
        predict_forces = self.model.get_forces(batch_data).detach().numpy()
        predict_stress = self.model.get_stresses(batch_data).detach().numpy()
        target_energy = batch_data['energy'].numpy()
        target_forces = batch_data['forces'].numpy()
        target_stress = batch_data['stress'].numpy()
        mae_energies = np.mean(np.abs(predict_energy - target_energy))
        r2_energies = 1 - np.sum((predict_energy - target_energy)**2) / \
            np.sum((target_energy - np.mean(target_energy))**2)
        mae_forces = np.mean(np.abs(predict_forces - target_forces))
        r2_forces = 1 - np.sum((predict_forces - target_forces)**2) / \
            np.sum((target_forces - np.mean(target_forces))**2)
        mae_stress = np.mean(np.abs(predict_stress - target_stress))
        r2_stress = 1 - np.sum((predict_stress - target_stress)**2) / \
            np.sum((target_stress - np.mean(target_stress))**2)
        return mae_energies, r2_energies, mae_forces, r2_forces, mae_stress, r2_stress

    def updatedataset(self, images):
        self.model.update_dataset(images)
        self.ani_model.update_dataset(images)

    def predict_energy(self, atoms, eval_std=False):
        batch_data = convert_frames([atoms], self.environment_provider)
        E, E_std = self.model.get_energies(batch_data, True)
        E, E_std = E.item(), E_std.item()
        if eval_std:
            return E, E_std
        else:
            return E

    def predict_forces(self, atoms):
        batch_data = convert_frames([atoms], self.environment_provider)
        F = self.model.get_forces(batch_data)
        return F.squeeze().detach().numpy()

    def predict_stress(self, atoms, eval_with_energy_std=False):
        batch_data = convert_frames([atoms], self.environment_provider)
        S = self.model.get_stresses(batch_data)
        return S.squeeze().detach().numpy()


frames = read('initpop1.traj', ':')
m = pytorchGPRmodel()
m.updatedataset(frames)
m.train(epoch1=1000, epoch2=30000)

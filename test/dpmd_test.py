from ase.io import read
from ani.model import ANI, GPR
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData, convert_frames
from torch.utils.data import DataLoader
from ani.kernel import RBF
import torch
from ani.utils import get_loss
from ani.cutoff import *
from ani.symmetry_functions import *


device = "cpu"
cutoff1 = 4.8
cutoff2 = 5.0

frames = read('TiO2-new.traj', ':')
n_split = int(0.8 * len(frames))
elements = []
for atoms in frames:
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))
environment_provider = ASEEnvironment(cutoff2)
cut_fn = SmoothCosineCutoff(cutoff1, cutoff2)
# dpmd = Deepmd_radius(100, cut_fn)
dpmd = Deepmd_angular(100, cut_fn)
representation = CombinationRepresentation(dpmd)


model = ANI(representation, environment_provider)

bb = convert_frames(frames[:1], environment_provider)

# # n_split = 120
# #
# # train_data = AtomsData(frames[:n_split], environment_provider)
# # train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
# # test_data = AtomsData(frames[n_split:], environment_provider)
# # test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=_collate_aseatoms)
# #
# # loss_fn = torch.nn.MSELoss()
# #
# kern = RBF(dimension=representation.dimension)
# # kern = RBF(dimension=1)
# model = GPR(representation, kern, environment_provider)
#
# model.update_dataset(frames[:n_split])
#
# # X = model.X.detach().numpy()
# # y = model.y.detach().numpy()
# # from sklearn.gaussian_process import GaussianProcessRegressor
# # from sklearn.gaussian_process.kernels import RBF, ConstantKernel
# # kernel = RBF(0.5, (1e-4, 10))
# # m = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
# # m.fit(X[:130], y[:130])
# # yy = m.predict(X[130:])
# # import matplotlib.pyplot as plt
# # plt.scatter(y[130:],yy)
#
# batch_data = convert_frames(frames[n_split:n_split+10], environment_provider)
# model.train(500)
# loss = get_loss(model, batch_data, verbose=True)
# print(loss)
# torch.save(model.state_dict(), 'parameter-gpr-zer.pkl')

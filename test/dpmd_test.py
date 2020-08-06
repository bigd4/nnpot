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
frames = read('dataset.traj', ':')
n_split = int(0.8 * len(frames))
elements = []
for atoms in frames:
    for ele in atoms:
        elements.append(ele.number)
elements = tuple(set(elements))
environment_provider = ASEEnvironment(5.0)
dpmd_r = Deepmd_radius(100, SmoothCosineCutoff(4.8, 5.0))
dpmd_a = Deepmd_angular(100, SmoothCosineCutoff(2.8, 3.0))
# representation = CombinationRepresentation(dpmd_r, dpmd_a)
representation = CombinationRepresentation(dpmd_r)

model = ANI(representation, environment_provider)


train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=_collate_aseatoms)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch = 3
min_loss = 1000
for i in range(epoch):
    for i_batch, batch_data in enumerate(train_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss = get_loss(model, batch_data, weight=[1.0, 0.0, 0.0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i_batch, batch_data in enumerate(test_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss, energy_loss, force_loss, stress_loss = \
            get_loss(model, batch_data, verbose=True)
        print(i, loss.cpu().detach().numpy(),
              energy_loss.cpu().detach().numpy(),
              force_loss.cpu().detach().numpy(),
              stress_loss.cpu().detach().numpy())
        if loss.cpu().detach().numpy() < min_loss:
            min_loss = loss.cpu().detach().numpy()
            torch.save(model.state_dict(), 'parameter-dpmd.pkl')

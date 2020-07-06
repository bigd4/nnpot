import logging
from ase.io import read
from ani.jjnet import ANI
from ani.environment import ASEEnvironment
from ani.dataloader import _collate_aseatoms, AtomsData
from ani.train import Trainer
from torch.utils.data import DataLoader
import torch


logging.basicConfig(filename='log.txt', level=logging.DEBUG,
                    format="%(asctime)s %(message)s", datefmt='%H:%M:%S')

# logging.info('cuda:{}\ndevice:{}'.format(torch.cuda.is_available(), torch.cuda.get_device_name(0)))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = "cpu"
cutoff = 3.0
n_radius = 30
n_angular = 10

net = ANI(n_radius, n_angular, cutoff)
loss_calculator = torch.nn.MSELoss()
trainer = Trainer(net, loss_calculator)

frames = read('dataset.traj', ':')
n_split = 130
environment_provider = ASEEnvironment(cutoff)
train_data = AtomsData(frames[:n_split], environment_provider)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=_collate_aseatoms)
test_data = AtomsData(frames[n_split:], environment_provider)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=_collate_aseatoms)

epoch = 10
trainer.train(epoch, train_loader, test_loader, device)

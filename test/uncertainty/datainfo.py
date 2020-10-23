import matplotlib.pyplot as plt
from ase.io import read, write
import numpy as np


np.random.seed(1)
frames = read('sps_all.xyz', ':')
np.random.shuffle(frames)
write('test.traj', frames[4500:])
# e = [a.info['energy'] / len(a) for a in frames]
# rho = [a.get_volume() / len(a) for a in frames]
# plt.scatter(rho[:4500], e[:4500])
# plt.scatter(rho[4500:], e[4500:])

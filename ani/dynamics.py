import warnings
import numpy as np

from ase.optimize.optimize import Optimizer


class FIRE:
    def __init__(self, atoms, logfile='-', trajectory=None,
                 dt=0.1, maxstep=None, dt_max=1.0, Nmin=5,
                 finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1):

        self.dt_start = dt_start
        self.maxstep = maxstep or 0.1
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.N_delay = N_delay
        self.N_maxneg = N_maxneg


    def run(self, steps):
        self.initialize()
        for i in range(steps):
            self.step(i)

    def initialize(self, atoms, hyper_dim):
        N_atoms = len(atoms)
        self.r = np.zeros(N_atoms, 3 + hyper_dim)
        self.r[:, :3] = atoms.get_positions()
        self.symbols = atoms.get_atomic_numbers()
        self.v = np.zeros_like(self.r)
        self.a = self.a_start
        self.dt = self.dt_start
        self.N_pos = 0
        self.N_neg = 0

    def step(self, epoch):
        f = get_force(self.r, self.symbols)
        P = np.vdot(f, self.v)
        if P > 0.0:
            self.N_pos += 1
            self.N_neg = 0
            if self.N_pos > self.N_delay:
                self.dt = min(self.dt * self.finc, self.dt_max)
                self.a *= self.fa
        else:
            self.N_pos = 0
            self.N_neg += 1
            if self.N_neg > self.N_maxneg:
                return False
            if epoch > self.N_delay:
                self.dt = max(self.dt * self.fdec, self.dt_min)
                self.a = self.astart
            self.x -= 0.5 * self.dt * self.v
            self.v[:] *= 0.0

        self.v += self.dt * f / self.masses
        self.v = (1.0 - self.a) * self.v + \
                 self.a * f * np.linalg.norm(self.v) / np.linalg.norm(f)

        dr = self.dt * self.v
        norm_dr = np.linalg.norm(dr)
        if norm_dr > self.maxstep:
            dr *= self.maxstep / norm_dr
        self.r += dr

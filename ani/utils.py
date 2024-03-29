import torch
import torch.nn as nn
import numpy as np
from ani.cutoff import polynomial_cut
from math import factorial
from torch.nn import functional as F


def get_elements(frames):
    elements = []
    for atoms in frames:
        for ele in atoms:
            elements.append(ele.number)
    elements = tuple(set(elements))
    return elements


def get_statistic(frames, prior=None):
    energy_peratom = []
    for atoms in frames:
        energy = atoms.info['energy']
        if prior is not None:
            energy -= prior(frames)
        energy_peratom.append(energy / len(atoms))
    mean, std = np.mean(energy_peratom), np.std(energy_peratom)
    return mean, std


def get_loss(model, batch_data,  weight=[1.0, 1.0, 1.0], verbose=False):
    w_energy, w_forces, w_stress = weight
    loss, energy_loss, force_loss, stress_loss = torch.zeros(4).to(batch_data['positions'])
    if w_energy > 0.:
        predict_energy = model.get_energies(batch_data) / batch_data['n_atoms']
        target_energy = batch_data['energy'] / batch_data['n_atoms']
        energy_loss = torch.mean((predict_energy - target_energy) ** 2)

    if w_forces > 0.:
        predict_forces = model.get_forces(batch_data)
        target_forces = batch_data['forces']
        force_loss = torch.mean(torch.sum(
            (predict_forces - target_forces) ** 2, 1) / batch_data['n_atoms'].unsqueeze(-1))

    if w_stress > 0.:
        predict_stress = model.get_stresses(batch_data)
        target_stress = batch_data['stress']
        stress_loss = torch.mean((predict_stress - target_stress) ** 2)

    loss += w_energy * energy_loss + w_forces * force_loss + w_stress * stress_loss
    if verbose:
        return loss, energy_loss, force_loss, stress_loss
    return loss


class PositiveParameter(nn.Parameter):
    def get(self):
        # return self.clamp(min=0)
        return self.clamp(min=0) + torch.log(1 + torch.exp(self.clamp(max=0))) * 2


class PositiveParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True,a=0.):
        new = super(PositiveParameter, cls).__new__(cls, data, requires_grad)
        new.a = a
        return new

    def get(self):
        return self.a + self.clamp(min=0) + torch.log(1 + torch.exp(self.clamp(max=0))) * 2


class BoundParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, a=0., b=1.):
        data = torch.log((data - a) / (b - data))
        new = super(BoundParameter, cls).__new__(cls, data, requires_grad)
        new.a = a
        new.b = b
        return new

    def get(self):
        return (self.b - self.a) * torch.sigmoid(self) + self.a


def legendre(l_max):
    coef = np.zeros((l_max+1, l_max+1))
    for l in range(l_max+1):
        for k in range(l % 2, l + 1, 2):
            coef[l, k] = (-1)**int((l-k)/2)/2**l*factorial(l+k)\
                         / (factorial(k)*factorial((l-k)//2)*factorial((l+k)//2))
        coef[l] /= np.sqrt(2*l+1) # const in sum of spherical harmonic
    return coef


def get_zernike_combination(n_max, l_max, diag):
    if not l_max:
        l_max = n_max
    assert l_max <= n_max
    if diag:
        subs = np.array([[n, n, l] for n in range(n_max + 1)
                         for l in range(min([n, l_max - (n - l_max) % 2]), -1, -2)])
    else:
        subs = np.array([[n1, n2, l] for n1 in range(n_max + 1) for n2 in range(n1 % 2, n1 + 1, 2)
                         for l in range(min([n2, l_max - (n2 - l_max) % 2]), -1, -2)])
    return subs.T


def zernike(n_max):
    coef = np.zeros((n_max+1, n_max+1, n_max+1))
    for n in range(n_max+1):
        for l in range(n, -1, -2):
            if l == n:
                coef[n, l, n] = 1
            elif l == n-2:
                if n >= 2:
                    coef[n, l] = (n+0.5) * coef[n, n] - (n-0.5) * coef[n-2, n-2]
                else:
                    coef[n, l] = (n+0.5) * coef[n, n]
            elif l == 0:
                n2 = 2*n
                M1 = (n2+1)*(n2-1)/(n+l+1)/(n-l)
                M2 = -0.5*((2*l+1)**2*(n2-1) + (n2+1)*(n2-1)*(n2-3))/(n+l+1)/(n-l)/(n2-3)
                M3 = -1*(n2+1)*(n+l-1)*(n-l-2)/(n+l+1)/(n-l)/(n2-3)
                coef[n, l] = np.convolve([M2, 0, M1], coef[n-2, l])[:n_max+1] + M3 * coef[(n - 4, l)]
            else:
                L1 = (2*n+1)/(n+l+1)
                L2 = -1*(n-l)/(n+l+1)
                coef[n, l] = np.convolve([0, L1], coef[(n-1, l-1)])[:n_max+1] + L2 * coef[(n-2, l)]
    # const in sum of spherical harmonic and cutoff
    for n in range(n_max+1):
        for l in range(n, -1, -2):
            coef[n, l] *= np.sqrt((2*n+3)/(2*l+1))
    return coef


def cut_zernike(n_max, n_cut=2):
    zernike_coef = zernike(n_max)
    cut_coef = polynomial_cut(n_cut)
    coef = np.zeros((n_max + 1, n_max + 1, n_max + n_cut +2))
    for n in range(n_max+1):
        for l in range(n, -1, -2):
            coef[n, l] = np.convolve(zernike_coef[n, l], cut_coef)
    return coef


class Polynomial(nn.Module):
    def __init__(self, coef):
        super(Polynomial, self).__init__()
        # self.coef = coef
        self.register_buffer("coef", coef)
        self.n_max = coef.size()[-1]

    def forward(self, inputs):
        inputs = inputs.to(self.coef.device)
        inputs = inputs.unsqueeze(-1)
        xx = torch.cat([inputs ** i for i in range(self.n_max)], -1)
        for _ in range(len(self.coef.size())-1):
            xx = xx.unsqueeze(-2)
        coef = self.coef
        for _ in range(len(inputs.size())-1):
            coef = coef.unsqueeze(0)
        return torch.sum(coef * xx, -1)


class OneHotEmbedding(nn.Module):
    def __init__(self, elements, trainable=False):
        super(OneHotEmbedding, self).__init__()
        max_elements = max(elements)
        n_elements = len(elements)
        weights = torch.zeros(max_elements + 1, n_elements)
        for idx, Z in enumerate(elements):
            weights[Z, idx] = 1.
        self.z_weights = nn.Embedding(max_elements + 1, n_elements)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False

    def forward(self, inputs):
        return self.z_weights(inputs)


class AtomicNumberEmbedding(nn.Module):
    def __init__(self, elements, trainable=False):
        super(AtomicNumberEmbedding, self).__init__()
        max_elements = max(elements)
        weights = torch.arange(max_elements + 1)[:, None].float()
        self.z_weights = nn.Embedding(max_elements + 1, 1)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False

    def forward(self, inputs):
        return self.z_weights(inputs)


def augment_data(frames, n, dx=0.1):
    add_frames = []
    for _ in range(n):
        atoms = frames[np.random.randint(len(frames))].copy()
        d_positions = np.random.rand(*atoms.positions.shape) * dx
        d_E = np.sum(d_positions * atoms.info['forces'])
        atoms.info['energy'] -= d_E
        atoms.positions += d_positions
        add_frames.append(atoms)
    frames.extend(add_frames)
    np.random.shuffle(frames)
    return frames


def shifted_softplus(x):
    return F.softplus(x) - np.log(2.0)

activation_dict = {'shifted_softplus': shifted_softplus}

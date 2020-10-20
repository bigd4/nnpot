import torch
import numpy as np


# TODO: Combine kalman filter with torch.optim.optimize

class KalmanFilter:
    """
    params: iter parameters
    h: function to get observable from state
    z: true observable

    torch float always imprecise, use np.matrix instead
    """
    def __init__(self, params, h, z,
                 epsilon=1e-3,
                 eta_0=1e-2, eta_tau=50, eta_max=1e-6,
                 q_0=1e-2, q_tau=50, q_min=1e-6,
                 ):
        self.params = [p for p in params if p.requires_grad]
        self.h = h
        self.z = z
        self.n_allp = int(sum([p.numel() for p in self.params]))
        self.P = np.eye(self.n_allp) / epsilon
        self.eta_0, self.eta_max, self.eta_tau = eta_0, eta_max, eta_tau
        self.q_0, self.q_min, self.q_tau = q_0, q_min, q_tau
        self.epoch = 0

    @property
    def Q(self):
        q = min(self.q_0 * np.exp(-self.epoch / self.q_tau), self.q_min)
        return q * np.eye(self.n_allp)

    @property
    def eta(self):
        eta = max(self.eta_0 * np.exp(-self.epoch / self.eta_tau), self.eta_max)
        return eta

    def step(self, data_loader):
        self.epoch += 1
        for i_batch, batch_data in enumerate(data_loader):
            y_predict = self.h(batch_data)
            y_observe = self.z(batch_data)
            n_batch = y_observe.size()[0]
            H = torch.zeros((n_batch, self.n_allp))

            for i, y in enumerate(y_predict):
                j = 0
                y.backward(retain_graph=True)
                for p in self.params:
                    H[i][j: j + p.numel()] = p.grad.view(-1)
                    p.grad.detach_()
                    p.grad.zero_()
                    j += p.numel()

            H = H.detach().numpy()
            A = np.linalg.pinv(np.eye(n_batch) / self.eta + H @ self.P @ H.T)
            K = self.P @ H.T @ A
            residual = (y_observe - y_predict).detach().numpy()
            delta = torch.tensor(K @ residual).float()
            j = 0
            for p in self.params:
                p.data.add_(delta[j: j + p.numel()].view(p.size()))
                j += p.numel()

            self.P += self.Q - K @ H @ self.P

import torch.nn as nn
import torch
from ani.utils import *
from ani.kernel import *


class SparseGPR(nn.Module):
    def __init__(self, kern):
        super(SparseGPR, self).__init__()
        self.lamb = BoundParameter(torch.tensor(0.01), a=1e-5, b=100.)
        self.kern = kern
        self.optimizer = torch.optim.Adam(self.parameters())

    def train(self, epoch, log_epoch=500):
        for i in range(epoch):
            loss = self.compute_log_likelihood()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % log_epoch == 0:
                print(i, ':', loss.item())

        X_N, X_M, Y = self.X, self.X_M, self.Y
        K_M = self.kern.K(X_M, X_M)
        L_M = torch.cholesky(K_M, upper=False)
        K_MN = self.kern.K(X_M, X_N)
        K_N = self.kern.Kdiag(X_N)
        V_MN = torch.solve(K_MN, L_M)[0]
        ell = torch.sqrt(K_N - torch.sum(V_MN ** 2, 0) + self.lamb.get())

        V_MN /= ell
        Y /= ell.view(-1, 1)
        K_MN /= ell

        A_M = torch.eye(X_M.size()[0]) + V_MN @ V_MN.t()
        a_M = (K_MN @ Y).view(-1, 1)

        L_A = torch.cholesky(A_M, upper=False)
        self.L = (L_M @ L_A).detach()
        self.V = torch.solve(a_M, self.L)[0].detach()

    def compute_log_likelihood(self):
        X_N, X_M, Y = self.X, self.X_M, self.Y
        K_N = self.kern.Kdiag(X_N)
        K_M = self.kern.K(X_M, X_M)
        L_M = torch.cholesky(K_M, upper=False)
        K_MN = self.kern.K(X_M, X_N)
        V_MN, _ = torch.solve(K_MN, L_M)
        Diag_K = torch.diag(K_N - torch.sum(V_MN ** 2, 0) + self.lamb.get())
        K = Diag_K + V_MN.t() @ V_MN

        L = torch.cholesky(K, upper=False)
        V, _ = torch.solve(self.Y, L)
        V = V.squeeze(1)

        ret = 0.5 * self.Y.size(0) * torch.tensor(np.log(2 * np.pi))
        ret += torch.log(torch.diag(L)).sum()
        ret += 0.5 * (V ** 2).sum()
        return ret

    def predict(self, Xnew):
        Kx = self.kern.K(self.X_M, Xnew)
        A = torch.solve(Kx, self.L)[0]
        y = torch.mm(A.t(), self.V)
        return y


class GPR(nn.Module):
    def __init__(self, kern):
        super(GPR, self).__init__()
        self.lamb = BoundParameter(torch.tensor(0.01), a=1e-5, b=100.)
        self.kern = kern
        self.optimizer = torch.optim.Adam(self.parameters())

    def train(self, epoch, log_epoch=500):
        for i in range(epoch):
            loss = self.compute_log_likelihood()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % log_epoch == 0:
                print(i, ':', loss.item())

        K = self.kern.K(self.X, self.X) + torch.eye(self.X.size(0)) * self.lamb.get()
        self.L = torch.cholesky(K, upper=False).detach()
        self.V = torch.solve(self.Y, self.L)[0].detach()

    def compute_log_likelihood(self):
        K = self.kern.K(self.X, self.X) + torch.eye(self.X.size(0)) * self.lamb.get()
        L = torch.cholesky(K, upper=False)
        V = torch.solve(self.Y, L)[0]
        V = V.squeeze(1)

        ret = 0.5 * self.Y.size(0) * torch.tensor(np.log(2 * np.pi))
        ret += torch.log(torch.diag(L)).sum()
        ret += 0.5 * (V ** 2).sum()
        return ret

    def predict(self, Xnew):
        Kx = self.kern.K(self.X, Xnew)
        A = torch.solve(Kx, self.L)[0]
        y = torch.mm(A.t(), self.V)
        return y

X = torch.randn((100, 5)) * 5
Y = (X ** 2).sum(1).view(-1, 1) + torch.randn((100, 1))*0.001
Xnew = torch.randn((10, 5))

kern = RBF()
model = SparseGPR(kern)
model.X = X
model.X_M = model.X[:100]
model.Y = Y
model.train(2000)

y1 = model.predict(Xnew)
y1_true = (Xnew ** 2).sum(1)

kern = RBF()
model2 = GPR(kern)
model2.X = X
model2.Y = Y
model2.train(2000)

y = model2.predict(Xnew)
y_true = (Xnew ** 2).sum(1)

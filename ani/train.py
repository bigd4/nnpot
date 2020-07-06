import torch
import logging


def get_loss(model, loss_fn, batch_data, weight=[1.0, 1.0, 1.0], verbose=False):
    w_energy, w_forces, w_stress = weight
    predict_energy = model(batch_data)
    predict_forces = -torch.autograd.grad(
        predict_energy.sum(),
        batch_data['positions'],
        create_graph=True,
        retain_graph=True
    )[0]
    predict_stress = torch.autograd.grad(
        predict_energy.sum(),
        batch_data['scaling'],
        create_graph=True,
        retain_graph=True
    )[0][:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / batch_data['volume']

    target_energy = batch_data['energy']
    target_forces = batch_data['forces']
    target_stress = batch_data['stress']
    energy_loss = loss_fn(predict_energy, target_energy)
    force_loss = loss_fn(predict_forces, target_forces)
    stress_loss = loss_fn(predict_stress, target_stress)

    loss = w_energy * energy_loss + w_forces * force_loss + w_stress * stress_loss
    if verbose:
        return loss, energy_loss, force_loss, stress_loss
    return loss


class Trainer:
    def __init__(self, model, loss_fn, weight=[1.0, 1.0, 1.0]):
        self.model = model
        self.loss_fn = loss_fn
        self.min_loss = 100
        self.weight = weight

    def get_loss(self, batch_data, verbose=False):
        return get_loss(self.model, self.loss_fn, batch_data, self.weight, verbose)

    def train(self, epoch, train_loader, test_loader, device):
        self.model.to(device)
        nn_parameters, hyper_parameters = [], []
        for key, value in self.model.named_parameters():
            if 'etas' in key or 'rss' in key:
                hyper_parameters.append(value)
            else:
                nn_parameters.append(value)

        nn_optimizer = torch.optim.Adam(nn_parameters)
        hyper_optimizer = torch.optim.Adam(hyper_parameters)

        for i in range(epoch):
            if i % 5 == 0:
                for i_batch, batch_data in enumerate(train_loader):
                    batch_data = {k: v.to(device) for k, v in batch_data.items()}
                    loss = self.get_loss(batch_data)
                    hyper_optimizer.zero_grad()
                    loss.backward()
                    hyper_optimizer.step()

            for i_batch, batch_data in enumerate(train_loader):
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                loss = self.get_loss(batch_data)
                nn_optimizer.zero_grad()
                loss.backward()
                nn_optimizer.step()

            for i_batch, batch_data in enumerate(test_loader):
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                loss, energy_loss, force_loss, stress_loss = \
                    self.get_loss(batch_data, True)
                logging.info('{}\t{}\t{}\t{}\t{}'.format(i, loss.cpu().detach().numpy(),
                                                         energy_loss.cpu().detach().numpy(),
                                                         force_loss.cpu().detach().numpy(),
                                                         stress_loss.cpu().detach().numpy()))
                if loss.cpu().detach().numpy() < self.min_loss:
                    self.min_loss = loss.cpu().detach().numpy()
                    torch.save(self.model.state_dict(), 'parameter.pkl')

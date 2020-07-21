import torch


def get_loss(model, batch_data, loss_fn=torch.nn.MSELoss(),  weight=[1.0, 1.0, 1.0], verbose=False):
    w_energy, w_forces, w_stress = weight
    predict_energy = model.get_energies(batch_data)
    predict_forces = model.get_forces(batch_data)
    predict_stress = model.get_stresses(batch_data)
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

import torch
import gpytorch
import botorch
import pickle as pkl

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model

def fit_gp_model(x, y, device='cpu'):
    """
    Fit GP model to data (x, y) and return the surrogate GP model
    """
    surrogate_gp_model = SingleTaskGP(x, y)
    if device == 'cuda':
        surrogate_gp_model = surrogate_gp_model.cuda()

    mll = ExactMarginalLogLikelihood(surrogate_gp_model.likelihood, surrogate_gp_model)
    fit_gpytorch_model(mll, optimizer=botorch.optim.fit.fit_gpytorch_torch)

    return surrogate_gp_model

def create_mean_var_dataset(points, values, size, traj_len, device='cpu'):
    means = []
    variances = []

    pt = torch.tensor(points, dtype=torch.float32).to(device)
    v = torch.tensor(values, dtype=torch.float32).to(device)

    for i in range(size):
        m = []
        var = []
        for j in range(traj_len):
            surrogate_model = fit_gp_model(pt[i, :j+1], v[i, :j+1].view(-1, 1))

            posterior = surrogate_model.posterior(pt[i, j+1:j+2])
            m.append(posterior.mean.detach())
            var.append(posterior.variance.detach())

        means.append(m)
        variances.append(var)

    return means, variances

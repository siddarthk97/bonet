import math
import torch
import gpytorch

def compute_gram_matrix(x_1, x_2, length_scale=1.):
    kernel = gpytorch.kernels.RBFKernel(length_scale=length_scale)

    return kernel(x_1, x_2)

def ucb_loss(preds, target_x, target_y, b=0.1, length_scale=1.):
    """
    preds: (batch, block_size, dim)
    target_x: (batch, block_size, dim)
    target_y: (batch, block_size, 1)
    """
    target_x = target_x.to(preds.device)
    # with torch.no_grad():
    gram_matrix = compute_gram_matrix(target_x, target_x, length_scale) # batch x block_size x block_size
    gram_matrix = gram_matrix.to(preds.device)
    q = gram_matrix.evaluate()

    k_ = compute_gram_matrix(preds, target_x, length_scale) # batch x block_size x block_size
    k_ = k_.to(preds.device)
    k_ = k_.evaluate()
    k_ = torch.tril(k_, diagonal=-1) # batch x block_size x block_size

    u = torch.linalg.solve(q, target_y)
    mu = k_ @ u
    sigma = -k_ @ torch.linalg.solve(q, k_.transpose(dim0=1, dim1=2))
    sigma = torch.diagonal(sigma, dim1=1, dim2=2) # batch x block_size
    sigma = sigma.unsqueeze(-1)
    # add k(x*, x*) term (always 1 for RBF)
    sigma = sigma + 1

    ucb = -(mu + math.sqrt(b) * torch.sqrt(sigma))
    loss = torch.sum(ucb, dim=1)
    return loss

if __name__ == '__main__':
    x = torch.arange(2*3*2).view(2,3,2)
    y = torch.arange(2*3*2).view(2,3,2)
    z = torch.arange(2*3).view(2,3,1)
    z = z.float()
    x = x.float()
    y = y.float()

    losses = ucb_loss(x, y, z)
    loss = torch.mean(losses)

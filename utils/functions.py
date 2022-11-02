import torch
from botorch.test_functions.synthetic import Branin

def quadratic(bias):
    return lambda x : -(x - bias) ** 2

def neg_branin(x):
    return -Branin().evaluate_true(torch.Tensor(x))

def scaled_shifted_neg_branin(x, scale, shift):
    assert scale > 0

    branin = Branin()
    x = torch.tensor(x)
    shift = torch.tensor(shift)

    return -scale * branin.evaluate_true(x - shift)

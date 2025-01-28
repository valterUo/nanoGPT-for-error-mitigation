import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias and explicit initialization. """

    def __init__(self, ndim, bias, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, dtype=config.precision))
        self.bias = nn.Parameter(torch.zeros(ndim, dtype=config.precision)) if bias else None
        self.eps = 1e-5

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
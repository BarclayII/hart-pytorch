
import torch as T
from torch import nn
import torch.nn.functional as F

class AdaptiveLoss(nn.Module):
    def __init__(self, n_losses=None, weights=None):
        self.lambdas = nn.ParameterList()
        if weights is not None:
            for w in weights:
                p = nn.Parameter(T.Tensor([w]))
                self.lambdas.append(p)
        else:
            for i in range(n_losses):
                p = nn.Parameter(T.Tensor([1]))
                self.lambdas.append(p)

    def forward(self, *losses):
        total_loss = sum(l * loss for l, loss in zip(self.lambdas, losses))
        reg = sum(l.log() for l in self.lambdas)
        return total_loss + reg

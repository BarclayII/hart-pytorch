
import torch as T
from torch import nn
import torch.nn.functional as F

class AdaptiveLoss(nn.Module):
    def __init__(self, n_losses=None, weights=None):
        nn.Module.__init__(self)
        self.lambdas = nn.ParameterList()
        if weights is not None:
            for w in weights:
                p = nn.Parameter(self.invtransform(T.Tensor([w])))
                self.lambdas.append(p)
        else:
            for i in range(n_losses):
                p = nn.Parameter(self.invtransform(T.Tensor([1])))
                self.lambdas.append(p)

    def transform(self, x):
        return x ** 2 + 1e-8

    def invtransform(self, x):
        return x.sqrt()

    def forward(self, *losses):
        lambdas = [self.transform(l) for l in self.lambdas]
        total_loss = sum(l * loss for l, loss in zip(lambdas, losses))
        # R(\lambda) in the paper is very weird.
        # They say -\sum_i log(1 / \lambda_i).  In this case, I can simply set
        # all \lambda_i's to be 0, so R(\lambda) is minus infinity and all the
        # other loss terms are 0.
        # Checked their code and it seems that they don't have the negative
        # sign - which makes more sense.
        reg = -sum(l.log() for l in lambdas)
        return total_loss + reg

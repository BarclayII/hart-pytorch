
import torch as T
from torch import nn
import torch.nn.functional as F
from util import *


def zoneout(x_old, x_new, p=0.5, training=True):
    '''
    p: drop probability
    '''
    if p == 0 or not training:
        return x_new
    assert 0 < p < 1
    assert x_old.size() == x_new.size()

    if training:
        noise = x_old.data.new()
        noise.bernoulli_(p)
        noise = T.autograd.Variable(noise.byte())

        output = x_new.masked_scatter(noise, x_old.masked_select(noise))
    else:
        output = (1 - p) * x_new + p * x_old

    return output


class Zoneout(nn.Module):
    def __init__(self, p=0.5):
        nn.Module.__init__(self)
        self.p = p

    def forward(self, x_old, x_new):
        return zoneout(x_old, x_new, self.p, self.training)


class ZoneoutLSTMCell(nn.Module):
    '''
    The variation of zoneout which reuses input dropout mask
    p: drop probability
    '''
    def __init__(self, input_size, hidden_size, bias=True, p=0.5):
        nn.Module.__init__(self)
        self.p = p
        self.W_gates = nn.Parameter(
                T.randn(input_size + hidden_size, 4 * hidden_size) * 0.1)
        self._hidden_size = hidden_size
        if bias:
            self.b_gates = nn.Parameter(T.zeros(4 * hidden_size))

    def forward(self, x, state):
        batch_size = x.size()[0]
        h_, c_, o_ = state
        xh = T.cat([h_, x], 1)
        ifog = xh @ self.W_gates + self.b_gates.unsqueeze(0)
        ifog = ifog.view(batch_size, 4, self._hidden_size)
        i, f, o = T.unbind(F.sigmoid(ifog[:, :3]), 1)
        g = ifog[:, 3].tanh()

        if self.training:
            d = i.data.new(i.size())
            d.bernoulli_(1 - self.p)
            d = T.autograd.Variable(d.float())

            c = f * c_ + d * i * g
            h = ((1 - d) * o + d * o_) * c.tanh()
        else:
            c = f * c_ + (1 - self.p) * i * g
            h = (self.p * o + (1 - self.p) * o_) * c.tanh()

        return h, (h, c, o)

    def zero_state(self, batch_size):
        return (
                tovar(T.zeros(batch_size, self._hidden_size)),
                tovar(T.zeros(batch_size, self._hidden_size)),
                tovar(T.ones(batch_size, self._hidden_size)),
                )

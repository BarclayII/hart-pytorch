
import torch as T
from torch import nn
import torch.nn.functional as F
from torchvision import models as tvmodels
from util import *
import numpy as np

class AlexNetModel(nn.Module):
    def __init__(self,
                 n_out_feature_maps=10,
                 layer=7,
                 upsample=False,
                 normlayer='batch',
                 keep_prob=.75):
        nn.Module.__init__(self)
        self.layer = layer
        self.upsample = upsample
        self.normlayer = normlayer
        self.keep_prob = keep_prob
        self.n_out_feature_maps = n_out_feature_maps

        alexnet = tvmodels.alexnet(pretrained=True)
        features = []
        for i, module in enumerate(alexnet.features):
            if i >= layer:
                break
            if isinstance(module, nn.MaxPool2d) and not upsample:
                continue
            features.append(module)
            if isinstance(module, nn.Conv2d):
                in_channels = module.out_channels
                self.n_out_channels = in_channels

        self.features = nn.Sequential(*features)

        self.readout = nn.Conv2d(in_channels, n_out_feature_maps, (1, 1))
        self.dropout = nn.Dropout(1 - keep_prob)
        if normlayer == 'batch':
            self.norm = nn.BatchNorm2d(n_out_feature_maps)
        else:
            self.norm = None

    def compute_output_size(self, input_size):
        output_size = input_size
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                output_size = conv_output_size(
                        output_size,
                        module.kernel_size,
                        module.padding,
                        module.stride)
        return output_size

    def compute_output_flatsize(self, input_size):
        return np.asscalar(np.prod(self.compute_output_size(input_size)))

    def forward(self, x):
        if self.upsample:
            raise NotImplementedError
        orig_output = self.features(x)
        readout = self.readout(orig_output)
        dropped = self.dropout(readout)
        normed = self.norm(dropped) if self.norm is not None else dropped
        return orig_output, readout, normed

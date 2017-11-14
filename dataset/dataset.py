
from torch.utils.data import Dataset, DataLoader
import os
from collections import namedtuple, OrderedDict
import numpy as np
import cv2
from util import *

# Decorator.  Resizes images and bboxes to the given size.
# In the dataset subclasses you only need to return a list of original
# RGB images and bboxes.
def resize_and_normalize(getitem):
    def new_getitem(self, i):
        images, bboxes, seqlen = getitem(self, i)

        for i in range(seqlen):
            rows, cols, _ = images[i].shape
            images[i] = cv2.resize(images[i], (self._cols, self._rows))
            if self._normalize:
                images[i] = torch_normalize_image(images[i])
            images[i] = images[i].transpose(2, 0, 1)

            bboxes[i][0] *= self._cols / cols
            bboxes[i][1] *= self._rows / rows
            bboxes[i][2] *= self._cols / cols
            bboxes[i][3] *= self._rows / rows

        return np.array(images), np.array(bboxes), seqlen
    return new_getitem


class VideoDataset(Dataset):
    def __init__(self,
                 normalize=True,
                 rows=None,
                 cols=None,
                 seqlen=None):
        self._normalize = normalize
        self._rows = rows
        self._cols = cols
        self._seqlen = seqlen


class VideoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0, shuffle=True):
        DataLoader.__init__(self,
                            dataset,
                            batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            drop_last=True)

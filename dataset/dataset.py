
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
                 seqlen=None,
                 **kwargs):
        '''
        kwargs: whatever arguments passed into _build_bboxes()
        '''
        self._normalize = normalize
        self._rows = rows
        self._cols = cols
        self._seqlen = seqlen

        self._bboxes = self._build_bboxes(**kwargs)
        self._keys = list(self._bboxes.keys())

        if seqlen is not None:
            lengths = [b.shape[0] - seqlen + 1 for b in self._bboxes.values()]
        else:
            lengths = [1 for b in self._bboxes.values()]
        self._index_segments = np.cumsum([0] + lengths)[:-1]

    def _build_bboxes(self, *args, **kwargs):
        '''
        Should return an OrderedDict
        The keys are whatever keys you'd use for _locate(), and must have two
        attributes @start and @end.  When the dataset fetches a subsequence,
        it picks a key first, then either iterates from @start to @end
        (inclusive) if self.seqlen is None, or it picks a subsequence between
        @start and @end.
        The values are the bounding boxes as numpy arrays.
        '''
        raise NotImplementedError

    def _locate(self, key, i):
        '''
        Returns the file name of the i-th image (can be either 0 based or 1
        based, depending on how your _build_bboxes() work) of the image
        sequence with key @key.

        @i comes from the iteration from key.start to key.end in __getitem__().
        '''
        raise NotImplementedError

    def __len__(self):
        return len(self._bboxes)

    @resize_and_normalize
    def __getitem__(self, idx):
        key_index = np.searchsorted(self._index_segments, idx, side='right') - 1
        key_offset = idx - self._index_segments[key_index]
        key = self._keys[key_index]

        start = (key_offset if self._seqlen is not None else 0) + key.start
        end = (start + self._seqlen - 1) if self._seqlen is not None else key.end
        seqlen = end - start + 1

        images = []
        bboxes = []

        for bbox_idx, i in enumerate(range(start, end + 1)):
            filename = self._locate(key, i)
            bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.
            images.append(rgb)

            bbox = self._bboxes[key][bbox_idx + key_offset]
            bboxes.append(bbox)

        return images, bboxes, seqlen


class VideoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0, shuffle=True):
        DataLoader.__init__(self,
                            dataset,
                            batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            drop_last=True)

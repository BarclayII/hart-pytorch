
from torch.utils.data import Dataset, DataLoader
import os
from collections import namedtuple, OrderedDict
import numpy as np
import cv2
from util import *
from .dataset import VideoDataset, resize_and_normalize

DatasetKey = namedtuple('DatasetKey', ['person', 'scenario', 'action', 'start', 'end'])

actions = {
        1: 'boxing',
        2: 'handclapping',
        3: 'handwaving',
        4: 'jogging',
        5: 'running',
        6: 'walking',
        }

def key_to_subdir(key):
    return 'person%02d_%s_d%1d' % (
            key.person, actions[key.action], key.scenario)

class KTHDataset(VideoDataset):
    def __init__(self,
                 directory,
                 bbox_info_file,
                 normalize=True,
                 rows=120,
                 cols=160,
                 seqlen=30):
        VideoDataset.__init__(self, normalize, rows, cols, seqlen)
        bbox_info_f = open(bbox_info_file)
        # skip first 4 lines
        for _ in range(4):
            bbox_info_f.readline()

        self._dir = directory
        self._bboxes = OrderedDict()

        for line in bbox_info_f:
            person, scenario, action, seq, start, end, bboxes = \
                    line.strip().split(' ', 6)
            if int(action) not in [4, 5, 6]:
                continue

            key = DatasetKey(
                    person=int(person),
                    scenario=int(scenario),
                    action=int(action),
                    start=int(start),
                    end=int(end),
                    )

            coords = [int(x) for x in bboxes.split()]
            bboxes = []
            for i in range(0, len(coords), 4):
                ymin, xmin, ymax, xmax = coords[i:i+4]
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
                bboxes.append([cx, cy, w, h])
            self._bboxes[key] = np.array(bboxes)
            self._keys = list(self._bboxes.keys())

        if seqlen is not None:
            lengths = [b.shape[0] - seqlen + 1 for b in self._bboxes.values()]
        else:
            lengths = [1 for b in self._bboxes.values()]
        self._index_segments = np.cumsum([0] + lengths)[:-1]

    def __len__(self):
        return len(self._bboxes)

    @resize_and_normalize
    def __getitem__(self, idx):
        key_index = np.searchsorted(self._index_segments, idx, side='right') - 1
        key_offset = idx - self._index_segments[key_index]
        key = self._keys[key_index]
        dir_ = os.path.join(self._dir, key_to_subdir(key))

        start = (key_offset if self._seqlen is not None else 0) + key.start
        end = (start + self._seqlen - 1) if self._seqlen is not None else key.end
        seqlen = end - start + 1

        images = []
        bboxes = []

        for bbox_idx, i in enumerate(range(start, end + 1)):
            filename = os.path.join(dir_, 'frame_%d.jpg' % i)
            bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.
            images.append(rgb)

            bbox = self._bboxes[key][bbox_idx + key_offset]
            bboxes.append(bbox)

        return images, bboxes, seqlen

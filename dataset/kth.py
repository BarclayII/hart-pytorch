
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
                 directory=None,
                 bbox_info_file=None,
                 normalize=True,
                 rows=120,
                 cols=160,
                 seqlen=30):
        VideoDataset.__init__(
                self,
                normalize,
                rows,
                cols,
                seqlen,
                directory=directory,
                bbox_info_file=bbox_info_file)
        self._dir = directory

    def _build_bboxes(self, directory=None, bbox_info_file=None):
        bbox_info_f = open(bbox_info_file)
        # skip first 4 lines
        for _ in range(4):
            bbox_info_f.readline()

        bboxes = OrderedDict()

        for line in bbox_info_f:
            person, scenario, action, seq, start, end, bbox_str = \
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

            coords = [int(x) for x in bbox_str.split()]
            cur_bboxes = []
            for i in range(0, len(coords), 4):
                ymin, xmin, ymax, xmax = coords[i:i+4]
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
                cur_bboxes.append([cx, cy, w, h])
            bboxes[key] = np.array(cur_bboxes)

        return bboxes

    def _locate(self, key, i):
        return os.path.join(
                self._dir,
                key_to_subdir(key),
                'frame_%d.jpg' % i,
                )

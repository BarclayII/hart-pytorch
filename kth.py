
from torch.utils.data import Dataset, DataLoader
import os
from collections import namedtuple, OrderedDict
import numpy as np
import cv2

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

def torch_normalize_image(x):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return (x - mean) / std

def torch_unnormalize_image(x):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return x * std + mean

class KTHDataset(Dataset):
    def __init__(self,
                 directory,
                 bbox_info_file,
                 normalize=True,
                 rows=120,
                 cols=160,
                 seqlen=30):
        bbox_info_f = open(bbox_info_file)
        # skip first 4 lines
        for _ in range(4):
            bbox_info_f.readline()

        self._dir = directory
        self._bboxes = OrderedDict()
        self._normalize = normalize
        self._rows = rows
        self._cols = cols
        self._seqlen = seqlen

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

    def __getitem__(self, idx):
        key_index = np.searchsorted(self._index_segments, idx, side='right') - 1
        key_offset = idx - self._index_segments[key_index]
        key = self._keys[key_index]
        dir_ = os.path.join(self._dir, key_to_subdir(key))

        start = (key_offset if self._seqlen is not None else 0) + key.start
        end = (start + self._seqlen - 1) if self._seqlen is not None else key.end
        seqlen = end - start + 1

        images = np.zeros((seqlen, 3, self._rows, self._cols))
        bboxes = np.zeros((seqlen, 4))

        for bbox_idx, i in enumerate(range(start, end + 1)):
            filename = os.path.join(dir_, 'frame_%d.jpg' % i)
            bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.
            rgb = cv2.resize(rgb, (self._cols, self._rows))
            rows, cols, _ = rgb.shape
            if self._normalize:
                rgb = torch_normalize_image(rgb)
            rgb = rgb.transpose(2, 0, 1)

            images[bbox_idx] = rgb

            bbox = self._bboxes[key][bbox_idx + key_offset]
            bbox[0] = bbox[0] / cols * self._cols
            bbox[1] = bbox[1] / rows * self._rows
            bbox[2] = bbox[2] / cols * self._cols
            bbox[3] = bbox[3] / rows * self._rows

            bboxes[bbox_idx] = bbox

        return images, bboxes, seqlen


class KTHDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0):
        DataLoader.__init__(self,
                            dataset,
                            batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            drop_last=True)

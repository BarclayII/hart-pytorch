
from torch.utils.data import Dataset, DataLoader
import os
from collections import namedtuple, OrderedDict
import numpy as np
import cv2
from util import *
from .dataset import VideoDataset
from lxml import etree
import pickle
import logging
import sh

DatasetKey = namedtuple('DatasetKey', ['folder', 'trackid', 'start', 'end'])
default_parser = etree.XMLParser(remove_blank_text=True)

class ImagenetVIDDataset(VideoDataset):
    def __init__(self,
                 data_directory=None,
                 annotation_pkl_dir=None,
                 annotation_directory=None,
                 normalize=True,
                 rows=480,
                 cols=640,
                 seqlen=60):
        self._data_dir = data_directory
        self._anno_pkl_dir = annotation_pkl_dir

        if not os.path.exists(annotation_pkl_dir):
            sh.mkdir('-p', annotation_pkl_dir)
            self._preprocess_annotations(
                    annotation_directory, annotation_pkl_dir)

        # Build the bboxes after preprocessing, if needed
        VideoDataset.__init__(
                self,
                normalize,
                rows,
                cols,
                seqlen,
                annotation_pkl_dir=annotation_pkl_dir,
                )

    def _build_bboxes(self, annotation_pkl_dir=None):
        bboxes = OrderedDict()
        for fname in os.listdir(annotation_pkl_dir):
            if not fname.endswith('.pkl'):
                continue
            with open(os.path.join(annotation_pkl_dir, fname), 'rb') as f:
                anno_key, anno_bboxes = pickle.load(f)
            if (self._seqlen is not None and
                    anno_key.end - anno_key.start + 1 < self._seqlen):
                continue
            bboxes[anno_key] = anno_bboxes

        return bboxes

    def _locate(self, key, i):
        return os.path.join(
                self._data_dir,
                key.folder,
                '%06d.JPEG' % i)

    def _commit(self, key, value):
        pkl_path = os.path.join(self._anno_pkl_dir, '%d.pkl' % self._size)
        with open(pkl_path, 'wb') as f:
            pickle.dump((key, value), f)
        self._size += 1

    def _preprocess_annotations(self, annotation_directory, annotation_pkl_dir):
        self._size = 0
        logging.info('Preprocessing annotations...')
        bboxes = OrderedDict()

        for rootdir, subdirs, files in os.walk(annotation_directory):
            xmlfiles = [f for f in files if f.endswith('.xml')]
            updates = OrderedDict()
            for i in range(len(xmlfiles)):
                path = os.path.join(rootdir, '%06d.xml' % i)
                logging.info(path)

                doc = etree.parse(path)

                folder = doc.find('folder')
                if folder is None:
                    logging.info('Skip %s: empty folder tag' % path)
                    continue
                folder = folder.text.strip()

                update_set = set()

                for obj in doc.findall('object'):
                    trackid = obj.find('trackid')
                    if trackid is None:
                        continue
                    trackid = int(trackid.text)

                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                    xmax = int(bndbox.find('xmax').text.strip())
                    xmin = int(bndbox.find('xmin').text.strip())
                    ymax = int(bndbox.find('ymax').text.strip())
                    ymin = int(bndbox.find('ymin').text.strip())
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin

                    key = DatasetKey(
                            folder=folder, trackid=trackid, start=0, end=0)
                    if key not in updates:
                        updates[key] = []

                    if key in updates:
                        logging.info('Adding %s with [%d, %d, %d, %d]' %
                                (key, cx, cy, w, h))
                        updates[key].append([cx, cy, w, h])
                        update_set.add(key)

                update_keys = list(updates.keys())
                for key in update_keys:
                    if key not in update_set:
                        newkey = DatasetKey(
                                folder=key.folder,
                                trackid=key.trackid,
                                end=i-1,
                                start=i-len(updates[key]),
                                )
                        logging.info('Flushing %s with %d bboxes' %
                                (newkey, len(updates[key])))
                        self._commit(newkey, np.array(updates[key]))
                        del updates[key]

            for key in updates:
                newkey = DatasetKey(
                        folder=key.folder,
                        trackid=key.trackid,
                        end=len(xmlfiles)-1,
                        start=len(xmlfiles)-len(updates[key]),
                        )
                logging.info('Flushing %s with %d bboxes' %
                        (newkey, len(updates[key])))
                self._commit(newkey, np.array(updates[key]))

        logging.info('Done')

import pathlib
import pickle
import time
from functools import partial

import numpy as np


REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_dataset_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


class Dataset(object):
    NumPointFeatures = -1

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _read_data(self, query):

        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

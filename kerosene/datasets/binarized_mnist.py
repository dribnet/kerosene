# -*- coding: utf-8 -*-
import fuel.datasets
from .dataset import Dataset

class BinarizedMNIST(Dataset):
    basename = "binarized_mnist"
    default_sources=['features']
    class_for_filename_patch = fuel.datasets.BinarizedMNIST

    def build_data(self, sets, sources):
        return map(lambda s: fuel.datasets.BinarizedMNIST(which_sets=[s], sources=sources), sets)

def load_data(sets=None, sources=None, fuel_dir=False):
    return BinarizedMNIST().load_data(sets, sources, fuel_dir);

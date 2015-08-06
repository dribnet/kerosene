# -*- coding: utf-8 -*-
import fuel.datasets
from .dataset import Dataset

class CIFAR100(Dataset):
    basename = "cifar100"
    default_sources=['features', 'coarse_labels']
    class_for_filename_patch = fuel.datasets.CIFAR100

    def build_data(self, sets, sources):
        return map(lambda s: fuel.datasets.CIFAR100(which_sets=[s], sources=sources), sets)

def load_data(sets=None, sources=None, fuel_dir=False):
    return CIFAR100().load_data(sets, sources, fuel_dir);

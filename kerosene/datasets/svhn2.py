# -*- coding: utf-8 -*-
import fuel.datasets
from .dataset import Dataset

class SVHN2(Dataset):
    basename = "svhn_format_2"
    class_for_filename_patch = fuel.datasets.SVHN

    def build_data(self, sets, sources):
        return map(lambda s: fuel.datasets.SVHN(which_format=2, which_sets=[s], sources=sources), sets)

def load_data(sets=None, sources=None, fuel_dir=False):
    return SVHN2().load_data(sets, sources, fuel_dir);

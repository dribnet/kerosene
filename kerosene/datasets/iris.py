# -*- coding: utf-8 -*-
import fuel.datasets
from .dataset import Dataset

class Iris(Dataset):
    basename = "iris"
    default_sets=['all']
    class_for_filename_patch = fuel.datasets.Iris

    def build_data(self, sets, sources):
        return map(lambda s: fuel.datasets.Iris(which_sets=[s], sources=sources), sets)

def load_data(sets=None, sources=None, fuel_dir=False):
    return Iris().load_data(sets, sources, fuel_dir);
